document.addEventListener("DOMContentLoaded", function() {
  const baseImages = ["0.png", "1.png", "2.png", "3.png", "4.png"];
  const imageWrapper = document.getElementById("image-wrapper");

  // Create image elements
  baseImages.forEach((src, index) => {
    const container = document.createElement('div');
    container.className = 'image-container';
    container.style.width = '20%';
    container.style.aspectRatio = "3/4";
    
    const baseImg = new Image();
    baseImg.src = "images/itw/rgb/"+ src;
    baseImg.className = 'base';
    container.appendChild(baseImg);

    const overlay1Img = new Image();
    overlay1Img.src = "images/itw/norm/"+ src;
    overlay1Img.className = 'overlay overlay1';
    container.appendChild(overlay1Img);

    const overlay2Img = new Image();
    overlay2Img.src = "images/itw/unc/"+ src;
    overlay2Img.className = 'overlay overlay2';
    overlay2Img.style.display = 'none';
    container.appendChild(overlay2Img);

    imageWrapper.appendChild(container);
  });

  let currentOverlayClass = "overlay1";
  const toggleButton = document.getElementById("toggleButton");
  toggleButton.innerText = `Switch to uncertainty`;
    
  const slider = document.getElementById("slider");

  toggleButton.addEventListener("click", function() {
    const value = slider.value;
    document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => el.style.display = 'none');
    currentOverlayClass = currentOverlayClass === "overlay1" ? "overlay2" : "overlay1";
      document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => {
      el.style.display = 'block';
      el.style.clipPath = `inset(0 ${100 - value}% 0 0)`; // Update this line
    });
      
    toggleButton.innerText = `Switch to ${currentOverlayClass === "overlay1" ? "uncertainty" : "normals"}`;
  });

  slider.addEventListener("input", function(event) {
    const value = event.target.value;
    document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => {
      el.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    });
  });
});
