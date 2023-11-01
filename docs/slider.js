function createImageSliderSet(options) {
  const {
    id, baseDirectory, baseImages, directories, nOverlays, overlayNames,
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
      overlayImg.className = `overlay overlay${i}-${id}`;
      overlayImg.style.display = 'none';
      container.appendChild(overlayImg);

    }
    
    const transitionLine = document.createElement('div');
    transitionLine.className = `transition-line`;
    transitionLine.id = `transition-line-${id}`;
    container.appendChild(transitionLine);

    imageWrapper.appendChild(container);
    
  }
  
  let currentOverlayClass = "overlay0";
  
  for (let i = 0; i < nOverlays; i++) {
  const radioButton = document.createElement('input');
  radioButton.type = 'radio';
  radioButton.name = `overlay-${id}`;
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

document.querySelectorAll(`#${radioButtonsId} input[name="overlay-${id}"]`).forEach((radio) => {

  radio.addEventListener('change', function() {
    document.querySelectorAll(`.${currentOverlayClass}-${id}`).forEach(el => el.style.display = 'none');
    currentOverlayClass = this.value;
    document.querySelectorAll(`.${currentOverlayClass}-${id}`).forEach(el => {
      el.style.display = 'block';
      el.style.clipPath = `inset(0 ${100 - slider.value}% 0 0)`; // Update this line
    });
  });
});

  slider.addEventListener("input", function(event) {
    const value = event.target.value;
    document.querySelectorAll(`.${currentOverlayClass}-${id}`).forEach(el => {
      el.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    });
    
  document.querySelectorAll(`#transition-line-${id}`).forEach(el => {
    el.style.left = `${value}%`;
  });
  
  });

}

function animateSlider(id){
  let slider = document.getElementById(id);
  let duration = 5000; // ms
  let hold = 3000; // Hold at max
  let startTime = Date.now();
  let shouldAnimate = true;
  function anim() {
    if (!shouldAnimate) {
        return;
    }

    let elapsed = (Date.now() - startTime) % duration;
    var eitherSide = (duration - hold) / 2
    slider.value = Math.min(100 * elapsed / eitherSide, 100 * ( - (elapsed - eitherSide - hold)/eitherSide))
    
    let event = new Event("input", {
        'bubbles': true,
        'cancelable': true
    });
    slider.dispatchEvent(event);

    requestAnimationFrame(anim);
    }
    
  slider.addEventListener("mousedown", function() {
    shouldAnimate =false;
  });
    
  anim()
  
}


document.addEventListener("DOMContentLoaded", function() {
  createImageSliderSet({
    id:"normals",
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
    id:"synth",
    baseDirectory: "images/synth",
    baseImages: ["0.png", "1.png", "2.png", "3.png", "4.png"],
    directories: ["norm", "mask", "keypoints"],
    nOverlays: 3,
    overlayNames: ["Normals", "Mask", "Keypoints"],
    imageWrapperId: "image-wrapper-synth",
    radioButtonsId: "radio-buttons-synth",
    sliderId: "slider-synth"
  });
  
  animateSlider("slider-synth")
  animateSlider("slider-normals")
  
});
