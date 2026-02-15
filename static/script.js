const liveVideo = document.getElementById("liveVideo");
const recordedVideo = document.getElementById("recordedVideo");
const recordBtn = document.getElementById("recordBtn");
const progressBar = document.getElementById("progressBar");
const resultText = document.getElementById("result");

let mediaRecorder;
let chunks = [];

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
.then(stream => {
    liveVideo.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => chunks.push(e.data);

    mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        chunks = [];

        // Show recorded video on right
        recordedVideo.src = URL.createObjectURL(blob);
        recordedVideo.style.display = "block";

        const file = new File([blob], "recorded.webm", { type: "video/webm" });

        uploadVideo(file);
    }
})
.catch(err => {
    alert("Could not access webcam: " + err);
});

// Record 2 seconds
recordBtn.onclick = () => {
    if (!mediaRecorder) return;

    resultText.innerText = "Recording...";
    progressBar.style.width = "0%";
    mediaRecorder.start();

    let start = Date.now();
    const interval = setInterval(() => {
        let elapsed = Date.now() - start;
        let percent = Math.min((elapsed / 2000) * 100, 100);
        progressBar.style.width = percent + "%";
    }, 50);

    setTimeout(() => {
        mediaRecorder.stop();
        clearInterval(interval);
        resultText.innerText = "Predicting...";
    }, 2000);
};

// Upload video to Flask backend
function uploadVideo(file) {
    const formData = new FormData();
    formData.append("video", file);

    fetch("/predict", { method: "POST", body: formData })
    .then(res => res.json())
    .then(data => {
        if (data.error) resultText.innerText = data.error;
        else resultText.innerText = `Prediction: ${data.prediction} (${data.confidence}%)`;
    })
    .catch(() => resultText.innerText = "Prediction failed");
}
