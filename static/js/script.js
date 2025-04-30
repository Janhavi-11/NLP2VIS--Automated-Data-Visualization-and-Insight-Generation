document.addEventListener("DOMContentLoaded", function () {
    let images = document.querySelectorAll('.chart-container img');
    let queries = [
        "Generate a bar graph for students opted for different courses",
        "Display a pie chart for students' CGPA distribution in college",
        "Generate a bar graph of the number of males and females in the engineering college"
    ];
    let queryText = document.querySelector('.query-text');
    let index = 0;

    // Ensure the first image is visible
    images[index].classList.add('active');

    setInterval(() => {
        images[index].classList.remove('active');
        index = (index + 1) % images.length;
        images[index].classList.add('active');
        queryText.textContent = "Query: " + queries[index];
    }, 3000);
});
