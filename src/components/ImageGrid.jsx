import { blurHashToDataURL } from '../utils.js'

export function ImageGrid({ images, setCurrentImage }) {
    return (
        <div>
            {images && images.map(({ id, url, ar, blur }) => (
                <div
                    key={id}
                    href={`https://unsplash.com/photos/${id}`}
                    onClick={() => {
                        // setCurrentImage({ id, url, ar, blur });
                    }}
                >
                    <img
                        alt=''
                        style={{ transform: 'translate3d(0, 0, 0)' }}
                        placeholder="blur"
                        blurDataURL={blurHashToDataURL(blur)}
                        src={`https://images.unsplash.com/${url}?auto=format&fit=crop&w=480&q=80`}
                        width={480}
                        height={480 / ar}
                    />
                </div>
            ))}
        </div>)
}