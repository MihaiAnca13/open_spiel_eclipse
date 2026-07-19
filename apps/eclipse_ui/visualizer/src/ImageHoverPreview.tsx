import type { ImagePreview } from './hooks/useImageHoverPreview';

export function ImageHoverPreview({ preview }: { preview: ImagePreview | null }) {
  if (!preview) return null;

  return (
    <div className="image-hover-preview" aria-hidden="true">
      <div className="image-hover-preview-frame">
        <img className="image-hover-preview-img" src={preview.src} alt="" />
        <div className="image-hover-preview-label">{preview.label}</div>
      </div>
    </div>
  );
}
