document.addEventListener('DOMContentLoaded', function() {
    // Add any interactive functionality if needed
    
    // Example: Highlight grid cells on hover
    const gridCells = document.querySelectorAll('.grid-cell');
    gridCells.forEach(cell => {
        cell.addEventListener('mouseover', function() {
            this.style.transform = 'scale(1.2)';
            this.style.transition = 'transform 0.2s';
        });
        
        cell.addEventListener('mouseout', function() {
            this.style.transform = 'scale(1)';
        });
    });
});