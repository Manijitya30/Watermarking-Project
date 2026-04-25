import { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div style={{ padding: 40 }}>
      <h2>Forgery Detection</h2>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload}>Upload</button>

      {result && (
        <div style={{ marginTop: 20 }}>
          <p>Prediction: {result.prediction}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}