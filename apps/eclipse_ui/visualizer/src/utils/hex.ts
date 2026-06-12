export const hexSize = 35;
export const IMAGE_ROTATION_OFFSET = 0;
export const SQRT3 = Math.sqrt(3);

// Axial (q,r) → pixel for a flat-top hex layout.
export function axialToPixel(q: number, r: number) {
  return {
    cx: hexSize * (1.5 * q),
    cy: hexSize * (SQRT3 / 2) * q + hexSize * SQRT3 * r,
  };
}

export function getHexPoints(cx: number, cy: number, r: number): string {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i); // flat-top: vertices left/right
    points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
  }
  return points.join(' ');
}

// Axial hex neighbor for direction 0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE (flat-top).
export function axialNeighbor(q: number, r: number, dir: number): [number, number] {
  const dirs: [number, number][] = [
    [1, 0],   // E
    [1, -1],  // NE
    [0, -1],  // NW
    [-1, 0],  // W
    [-1, 1],  // SW
    [0, 1],   // SE
  ];
  const [dq, dr] = dirs[dir] ?? [0, 0];
  return [q + dq, r + dr];
}
