let userVector = Array(50).fill(0);
let ratedMovies = [];
let currentMovie = null;
let step = 0;
const totalSteps = 20;

// Initialize study
document.addEventListener('DOMContentLoaded', () => {
    fetch('/get_movie')
        .then(res => res.json())
        .then(movie => {
            currentMovie = movie;
            displayMovie(movie);
            updatePredictions();
        });
    
    document.getElementById('rating-slider').addEventListener('input', updatePredictions);
});

function displayMovie(movie) {
    document.getElementById('movie-title').textContent = `${movie.title} (${movie.year})`;
    document.getElementById('movie-genres').textContent = movie.genres;
}

function updatePredictions() {
    const rating = parseFloat(document.getElementById('rating-slider').value);
    document.getElementById('rating-display').textContent = rating.toFixed(1);
    
    // Get predictions for different rating scenarios
    fetch('/get_recommendations', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            user_vector: userVector,
            movie_id: currentMovie.movieId,
            rating: rating
        })
    })
    .then(res => res.json())
    .then(recommendations => {
        displayRecommendations('high-rating-pred', recommendations);
    });
    
    // Repeat for low rating scenario
    // ...
}

function displayRecommendations(elementId, movies) {
    const ul = document.getElementById(elementId);
    ul.innerHTML = '';
    movies.forEach(movie => {
        const li = document.createElement('li');
        li.textContent = movie.title;
        ul.appendChild(li);
    });
}

document.getElementById('submit-rating').addEventListener('click', () => {
    const rating = parseFloat(document.getElementById('rating-slider').value);
    
    // Update user vector
    fetch('/update_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            user_vector: userVector,
            movie_id: currentMovie.movieId,
            rating: rating
        })
    });
    
    // Next step
    step++;
    if(step >= totalSteps) {
        window.location.href = '/results';
    } else {
        loadNextMovie();
    }
});