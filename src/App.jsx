import { useEffect } from "react";
import { useCallback } from "react";
import { useRef } from "react";
import { useState } from "react";
import { ImageGrid, SearchBar } from "./components";

function App() {
  const [ready, setReady] = useState(null);
  const [images, setImages] = useState(null);

  // Create a reference to the worker object.
  const worker = useRef(null);

  // We use the `useEffect` hook to set up the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
    }
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "initiate":
          setReady(false);
          break;
        case "ready":
          setReady(true);
          break;
        case "complete":
          setImages(e.data.output);
          break;
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () =>
      worker.current.removeEventListener("message", onMessageReceived);
  });

  const search = useCallback((text) => {
    if (worker.current) {
      worker.current.postMessage({ text });
    }
  }, []);

  return (
    <div>
      <SearchBar search={search} />
      {ready === false && <p>Loading model and database...</p>}
      <ImageGrid images={images} />
    </div>
  );
}

export default App;
