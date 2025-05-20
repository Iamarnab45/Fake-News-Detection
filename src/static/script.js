document.getElementById('newsForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const url = document.getElementById('newsUrl').value;
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    
    // Show loading spinner
    loading.classList.remove('d-none');
    result.classList.add('d-none');
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        // Update result display
        document.getElementById('prediction').textContent = data.is_fake ? 'FAKE NEWS' : 'REAL NEWS';
        document.getElementById('prediction').className = `prediction-text ${data.is_fake ? 'fake' : 'real'}`;
        document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('explanation').textContent = data.explanation;
        
        // Show results
        loading.classList.add('d-none');
        result.classList.remove('d-none');
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while analyzing the article. Please try again.');
        loading.classList.add('d-none');
    }
}); 