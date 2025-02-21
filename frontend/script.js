
function checkNews() {
    let text = document.getElementById("newsInput").value.trim(); // Trim whitespace

    // Check if input is empty
    if (!text) {
        document.getElementById("result").innerText = "Please enter some text!";
        return;
    }

    // Show loading state
    document.getElementById("result").innerText = "Checking...";
    /*fetch("http://localhost:5000/predict",   // ✅ Use full backend URL
        {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server error: " + response.status);
        }
        return response.json();
    })
    .then(data => {
        let resultText = data.prediction === "1" ? "Real News ✅" : "Fake News ❌";
        document.getElementById("result").innerText = "Result: " + resultText;
    })*/
        fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Response from Server:", data);  // ✅ Debugging line
            document.getElementById("result").innerText = "The news is  " + data.prediction;
            
        })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error checking news. Please try again.";
    });
}
