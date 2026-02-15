const liveVideo = document.getElementById("liveVideo");
const recordedVideo = document.getElementById("recordedVideo");
const recordBtn = document.getElementById("recordBtn");
const progressBar = document.getElementById("progressBar");
const resultText = document.getElementById("result");
const previewHint = document.getElementById("previewHint");

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
        const url = URL.createObjectURL(blob);
        recordedVideo.src = url;
        recordedVideo.style.display = "block";
        if (previewHint) previewHint.style.display = "none";
        
        // Create file and upload
        const file = new File([blob], "recorded_" + Date.now() + ".webm", { 
            type: "video/webm" 
        });

        uploadVideo(file);
        
        // Clean up URL after video is loaded
        recordedVideo.onloadeddata = () => {
            URL.revokeObjectURL(url);
        };
    };
})
.catch(err => {
    alert("Could not access webcam: " + err);
    console.error("Webcam error:", err);
});

// Record 2 seconds
recordBtn.onclick = () => {
    if (!mediaRecorder) {
        alert("Camera not initialized yet");
        return;
    }

    // Reset UI
    resultText.innerHTML = "Recording...";
    progressBar.style.width = "0%";
    recordBtn.disabled = true;
    recordBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Recording...';
    
    // Start recording
    chunks = [];
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
        resultText.innerHTML = "Predicting...";
        recordBtn.disabled = false;
        recordBtn.innerHTML = '<i class="fas fa-circle"></i> Record 2-sec Video';
    }, 2000);
};

// Upload video to Flask backend
function uploadVideo(file) {
    const formData = new FormData();
    formData.append("video", file);

    fetch("/predict", { 
        method: "POST", 
        body: formData 
    })
    .then(res => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        if (data.error) {
            resultText.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${data.error}`;
        } else {
            // Determine confidence color
            let confidenceColor = "#28a745";
            if (data.confidence < 70) confidenceColor = "#ffc107";
            if (data.confidence < 50) confidenceColor = "#dc3545";
            
            resultText.innerHTML = `
                <div style="font-size: 32px; color: #667eea; margin-bottom: 10px;">
                    ${data.prediction}
                </div>
                <div style="font-size: 18px;">
                    <i class="fas fa-check-circle" style="color: ${confidenceColor};"></i>
                    Confidence: ${data.confidence}%
                </div>
                <div style="font-size: 14px; color: #666; margin-top: 10px;">
                    (${data.english_label})
                </div>
            `;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultText.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Prediction failed: ${error.message}`;
    });
}

// Add keyboard support
document.addEventListener('keypress', (e) => {
    if (e.key === 'r' || e.key === 'R') {
        recordBtn.click();
    }
});