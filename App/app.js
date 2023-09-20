const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultsDiv = document.getElementById("results");

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
    video.srcObject = stream;
}

async function detectFrame() {
    ctx.drawImage(video, 0, 0, 640, 480);
    const frame = canvas.toDataURL('image/jpeg', 1.0);

    const formData = new FormData();
    formData.append("image", frame);

    fetch("/detect_mask", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultsDiv.innerHTML = "";

        data.results.forEach(result => {
            const resultDiv = document.createElement("div");
            resultDiv.innerHTML = `
                <p>Label: ${result.label}</p>
                <p>Probability: ${result.probability.toFixed(2)}</p>
                <p>Coordinates: startX=${result.coordinates.startX}, startY=${result.coordinates.startY}, endX=${result.coordinates.endX}, endY=${result.coordinates.endY}</p>
            `;
            results
