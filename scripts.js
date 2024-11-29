// Show the selected page
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId).classList.add('active');
}

// Handle predictions
function predict(diseaseType) {
    let form = document.getElementById(`${diseaseType}-form`);
    let formData = new FormData(form);
    let data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });

    fetch(`http://localhost:5000/predict/${diseaseType}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById(`${diseaseType}-result`).textContent = result.message;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
