document.addEventListener("DOMContentLoaded", function() {
  const baseImages = ["0.png", "1.png", "2.png", "3.png", "4.png"];
  const directories = ["norm", "unc"] // base + overlays
  
  const nImages = baseImages.length
  const nOverlays = 2;  // You can set n to any number
  const overlayNames = ["Normals", "Uncertainty"]
  
  const imageWrapper = document.getElementById("image-wrapper");
  
   
   // Create image elements
  for (let n = 0; n < nImages; n ++){
    const container = document.createElement('div');
    container.className = 'image-container';
    container.style.width = '20%';
    container.style.aspectRatio = "3/4";
    
    const src = baseImages[n];
    
    const baseImg = new Image();
    baseImg.src = "images/itw/rgb/"+ src;
    baseImg.className = 'base';
    container.appendChild(baseImg);

    for (let i = 0; i < nOverlays; i++) {
      const overlayImg = new Image();
      overlayImg.src = `images/itw/` + directories[i] + '/' + src;
      overlayImg.className = `overlay overlay${i}`;
      overlayImg.style.display = 'none';
      container.appendChild(overlayImg);

    }
    
    const transitionLine = document.createElement('div');
    transitionLine.className = 'transition-line';
    container.appendChild(transitionLine);

    imageWrapper.appendChild(container);
    
  }
  
  let currentOverlayClass = "overlay0";
  
  for (let i = 0; i < nOverlays; i++) {
  const radioButton = document.createElement('input');
  radioButton.type = 'radio';
  radioButton.name = 'overlay';
  radioButton.value = `overlay${i}`;
  radioButton.id = `radio${i}`;
  const label = document.createElement('label');
  label.htmlFor = `radio${i}`;
  label.innerText = overlayNames[i];
  document.getElementById('radio-buttons').appendChild(radioButton);
  document.getElementById('radio-buttons').appendChild(label);
}
  
  // set initial choice to be overlay0
setTimeout(() => {
  const defaultRadio = document.getElementById('radio0');
  defaultRadio.checked = true;
  const event = new Event('change', { 'bubbles': true, 'cancelable': true });
  defaultRadio.dispatchEvent(event);
}, 100);

  
  const slider = document.getElementById("slider");

document.querySelectorAll('input[name="overlay"]').forEach((radio) => {
  radio.addEventListener('change', function() {
    document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => el.style.display = 'none');
    currentOverlayClass = `overlay${this.id.replace("radio", "")}`;
    document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => {
      el.style.display = 'block';
      el.style.clipPath = `inset(0 ${100 - slider.value}% 0 0)`; // Update this line
    });
  });
});

  slider.addEventListener("input", function(event) {
    const value = event.target.value;
    document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => {
      el.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    });
    
  document.querySelectorAll('.transition-line').forEach(el => {
    el.style.left = `${value}%`;
  });
  
  });
});
