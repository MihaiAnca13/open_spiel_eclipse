# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ppo._ObsRows against the torch.cat reference it replaced.

_ObsRows exists to avoid materializing ``torch.cat([rollout_buffer, extras])``
during learn(). Every row it hands to the network must be the row that concat
would have produced -- and the two indexings it composes (the --league trainable
mask, then the whole-minibatches truncation) are exactly where getting this wrong
silently trains on mismatched (obs, action) pairs rather than crashing.

So the reference here IS the naive concat: build it, index it the obvious way,
and require equality through every combination of buffer device and dtype.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from open_spiel.python.pytorch.ppo import _ObsRows

SEED = 20260813
N_MAIN = 96
N_EXTRA = 7
OBS = 5

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fixture(device, dtype):
  """(rows, reference) for a fresh random main/extra pair.

  Values are small integers so fp16 storage is exact: this test is about row
  ADDRESSING, and letting rounding into it would only blur that signal.
  """
  g = torch.Generator().manual_seed(SEED)
  main = torch.randint(0, 512, (N_MAIN, OBS), generator=g).to(
      device=device, dtype=dtype)
  extra = torch.randint(0, 512, (N_EXTRA, OBS), generator=g).to(
      device=device, dtype=dtype)
  rows = _ObsRows(main, extra, out_dtype=torch.float32)
  reference = torch.cat([main, extra]).to(device="cpu", dtype=torch.float32)
  return rows, reference


class ObsRowsTest(parameterized.TestCase):

  @parameterized.product(device=_DEVICES,
                         dtype=[torch.float32, torch.float16])
  def test_minibatch_matches_concat(self, device, dtype):
    rows, reference = _fixture(device, dtype)
    self.assertLen(rows, N_MAIN + N_EXTRA)
    # A minibatch that straddles the main/extra boundary, in shuffled order --
    # the arrangement np.random.shuffle actually produces in _learn_core.
    mb = np.array([0, N_MAIN + 3, 40, N_MAIN, N_MAIN + N_EXTRA - 1, 95, 7])
    got = rows.minibatch(mb, torch.device("cpu"))
    self.assertEqual(got.dtype, torch.float32)
    torch.testing.assert_close(got, reference[mb])

  @parameterized.product(device=_DEVICES,
                         dtype=[torch.float32, torch.float16])
  def test_main_only_and_extra_only_minibatches(self, device, dtype):
    """The two fast paths that skip the scatter (all-main / all-extra)."""
    rows, reference = _fixture(device, dtype)
    for mb in (np.array([1, 2, 3]),
               np.arange(N_MAIN, N_MAIN + N_EXTRA)):
      torch.testing.assert_close(rows.minibatch(mb, torch.device("cpu")),
                                 reference[mb])

  @parameterized.product(device=_DEVICES,
                         dtype=[torch.float32, torch.float16])
  def test_league_mask_then_truncate(self, device, dtype):
    """select(bool mask) then select(slice), composed as _learn_core does.

    The mask lives on the TRAINING device (it comes from self.trainable), while
    the row ids live on the host -- indexing one with the other unconverted is
    the bug that broke --league when the buffer moved off the GPU.
    """
    rows, reference = _fixture(device, dtype)
    keep = torch.zeros(N_MAIN + N_EXTRA, dtype=torch.bool, device=device)
    keep[::3] = True            # a scattered "trainable" subset of main rows
    keep[N_MAIN:] = True        # extras are always trainable
    kept = rows.select(keep)
    ref_kept = reference[keep.cpu()]
    self.assertLen(kept, ref_kept.shape[0])

    # Whole-minibatches truncation on top of the filtered view.
    batch_size = (len(kept) // 4) * 4
    trimmed = kept.select(slice(None, batch_size))
    self.assertLen(trimmed, batch_size)

    mb = np.arange(batch_size)[::-1].copy()     # reversed, to catch id/position
    torch.testing.assert_close(trimmed.minibatch(mb, torch.device("cpu")),
                               ref_kept[:batch_size][mb])

  @parameterized.parameters(*_DEVICES)
  def test_no_extras(self, device):
    """n_extra == 0 must behave like a plain view of the rollout buffer."""
    g = torch.Generator().manual_seed(SEED)
    main = torch.randint(0, 512, (N_MAIN, OBS), generator=g).to(device)
    rows = _ObsRows(main, None, out_dtype=torch.float32)
    self.assertLen(rows, N_MAIN)
    mb = np.array([5, 0, N_MAIN - 1])
    torch.testing.assert_close(rows.minibatch(mb, torch.device("cpu")),
                               main[mb].to(device="cpu", dtype=torch.float32))
    # An empty extra tensor is the same case, and must not take the scatter path.
    rows_empty = _ObsRows(main, main.new_zeros((0, OBS)),
                          out_dtype=torch.float32)
    self.assertLen(rows_empty, N_MAIN)
    torch.testing.assert_close(rows_empty.minibatch(mb, torch.device("cpu")),
                               main[mb].to(device="cpu", dtype=torch.float32))


if __name__ == "__main__":
  torch.manual_seed(SEED)
  absltest.main()
