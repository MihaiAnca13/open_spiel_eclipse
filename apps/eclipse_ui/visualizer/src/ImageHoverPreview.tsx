import { useCallback, useEffect, useRef, useState } from 'react';

interface ImagePreview {
  src: string;
  label: string;
}

export function useImageHoverPreview(delayMs = 1000) {
  const timerRef = useRef<number | null>(null);
  const [preview, setPreview] = useState<ImagePreview | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const beginPreview = useCallback((nextPreview: ImagePreview) => {
    clearTimer();
    timerRef.current = window.setTimeout(() => {
      setPreview(nextPreview);
      timerRef.current = null;
    }, delayMs);
  }, [clearTimer, delayMs]);

  const clearPreview = useCallback(() => {
    clearTimer();
    setPreview(null);
  }, [clearTimer]);

  useEffect(() => clearTimer, [clearTimer]);

  return { preview, beginPreview, clearPreview };
}

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
