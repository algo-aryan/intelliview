// Site navbar mobile menu functionality
function toggleSiteMenu() {
    const navLinks = document.querySelector('.site-nav-links');
    const menuToggle = document.querySelector('.site-mobile-toggle');
    
    navLinks.classList.toggle('active');
    menuToggle.classList.toggle('active');
    
    // Animate hamburger lines
    const lines = menuToggle.querySelectorAll('.site-hamburger-line');
    if (navLinks.classList.contains('active')) {
        lines[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
        lines[1].style.opacity = '0';
        lines[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
    } else {
        lines[0].style.transform = 'none';
        lines[1].style.opacity = '1';
        lines[2].style.transform = 'none';
    }
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
    const navbar = document.querySelector('.site-navbar');
    const navLinks = document.querySelector('.site-nav-links');
    const menuToggle = document.querySelector('.site-mobile-toggle');
    
    if (!navbar.contains(event.target) && navLinks.classList.contains('active')) {
        navLinks.classList.remove('active');
        menuToggle.classList.remove('active');
        
        const lines = menuToggle.querySelectorAll('.site-hamburger-line');
        lines[0].style.transform = 'none';
        lines[1].style.opacity = '1';
        lines[2].style.transform = 'none';
    }
});

// Close mobile menu when window is resized to desktop
window.addEventListener('resize', function() {
    if (window.innerWidth > 768) {
        const navLinks = document.querySelector('.site-nav-links');
        const menuToggle = document.querySelector('.site-mobile-toggle');
        
        navLinks.classList.remove('active');
        menuToggle.classList.remove('active');
        
        const lines = menuToggle.querySelectorAll('.site-hamburger-line');
        lines[0].style.transform = 'none';
        lines[1].style.opacity = '1';
        lines[2].style.transform = 'none';
    }
});
