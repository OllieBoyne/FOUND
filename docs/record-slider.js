// Tool for creating animations. Attach this to HTML to force slider
let slider = document.getElementById("slider-normals");
let duration = 3000; // Duration in milliseconds (2 seconds)
let hold = 2000; // Hold at max
let startTime = Date.now();
let shouldAnimate = true;

function animateSlider() {
    if (shouldAnimate) {
        return;
    }

    let elapsed = (Date.now() - startTime) % duration;
    
    var eitherSide = (duration - hold) / 2
    
    progress = Math.min(100 * elapsed / eitherSide, 100 * ( - (elapsed - eitherSide - hold)/eitherSide))
    
    slider.value = progress
    
    let event = new Event("input", {
        'bubbles': true,
        'cancelable': true
    });
    slider.dispatchEvent(event);

    requestAnimationFrame(animateSlider);
}

// Uncomment to activate
// document.addEventListener("DOMContentLoaded", function() {
//     animateSlider();
// });
