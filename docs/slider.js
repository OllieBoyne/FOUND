function createImageSliderSet(options) {
  const {
    baseDirectory, baseImages, directories, nOverlays, overlayNames,
    imageWrapperId, radioButtonsId, sliderId 
  } = options;

  const nImages = baseImages.length;
  const imageWrapper = document.getElementById(imageWrapperId);

   // Create image elements
  for (let n = 0; n < nImages; n ++){
    const container = document.createElement('div');
    container.className = 'image-container';
    container.style.width = '20%';
    container.style.aspectRatio = "3/4";
    
    const src = baseImages[n];
    
    const baseImg = new Image();
    baseImg.src = baseDirectory + "/rgb/"+ src;
    baseImg.className = 'base';
    container.appendChild(baseImg);

    for (let i = 0; i < nOverlays; i++) {
      const overlayImg = new Image();
      overlayImg.src = baseDirectory + "/" + directories[i] + '/' + src;
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
  radioButton.id = `${radioButtonsId}${i}`;
  const label = document.createElement('label');
  label.htmlFor = `${radioButtonsId}${i}`;
  label.innerText = overlayNames[i];
  document.getElementById(radioButtonsId).appendChild(radioButton);
  document.getElementById(radioButtonsId).appendChild(label);
}
  
  // set initial choice to be overlay0
setTimeout(() => {
  const defaultRadio = document.getElementById(`${radioButtonsId}0`);
  defaultRadio.checked = true;
  const event = new Event('change', { 'bubbles': true, 'cancelable': true });
  defaultRadio.dispatchEvent(event);
}, 100);

  
  const slider = document.getElementById(sliderId);

document.querySelectorAll('input[name="overlay"]').forEach((radio) => {

  radio.addEventListener('change', function() {
    document.querySelectorAll(`.${currentOverlayClass}`).forEach(el => el.style.display = 'none');
    currentOverlayClass = this.value;
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

}


document.addEventListener("DOMContentLoaded", function() {
  createImageSliderSet({
    baseDirectory: "images/itw",
    baseImages: ["0.png", "1.png", "2.png", "3.png", "4.png"],
    directories: ["norm", "unc"],
    nOverlays: 2,
    overlayNames: ["Normals", "Uncertainty"],
    imageWrapperId: "image-wrapper-normals",
    radioButtonsId: "radio-buttons-normals",
    sliderId: "slider-normals"
  });

  createImageSliderSet({
    baseDirectory: "images/synth",
    baseImages: ["0.png", "1.png", "2.png", "3.png", "4.png"],
    directories: ["mask", "norm", "keypoints"],
    nOverlays: 3,
    overlayNames: ["Mask", "Normals", "Keypoints"],
    imageWrapperId: "image-wrapper-synth",
    radioButtonsId: "radio-buttons-synth",
    sliderId: "slider-synth"
  });
});
