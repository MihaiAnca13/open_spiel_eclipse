export const hexSize = 35;
// The PNG tile art is 180 degrees from the backend's rotation=0 wormhole
// masks. Apply that fixed offset, then apply the game's rotation.
export const IMAGE_ROTATION_OFFSET = 180;
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
