export function ImageGrid({ images, setCurrentImage }) {
  return (
    <div>
      {images &&
        images.map(({ id, url }) => (
          <div
            key={id}
            onClick={() => {
              // setCurrentImage({ id, url, ar, blur });
            }}
          >
            <img
              alt=""
              style={{ transform: "translate3d(0, 0, 0)" }}
              src={url}
              width={480}
            />
          </div>
        ))}
    </div>
  );
}
