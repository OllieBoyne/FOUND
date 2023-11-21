import torch
import numpy as np
import cv2

def seg_to_rgb(arr: np.ndarray):
	img = np.dstack([arr]*3)
	return img

def tens2rgb(tens: torch.Tensor):
	"""Tensor, convert to numpy, convert to 0-255"""
	if torch.is_tensor(tens):
		tens = tens.cpu().detach().numpy()

	if tens.max() <= 1: tens *= 255
	return tens

def reduce_ax(arr: np.ndarray, n=1):
	"""Reduce to last n dims by taking 0th element along all reduced dims"""
	while arr.ndim > n:
		arr = arr[0]
	return arr

def seg_overlap(seg1, seg2):
	"""Return image with Red channel = seg 1 only, Blue channel = Seg 2 only, Green channel = overlap"""
	if torch.is_tensor(seg1): seg1 = seg1.cpu().detach().numpy()
	if torch.is_tensor(seg2): seg2 = seg2.cpu().detach().numpy()

	H, W = seg1.shape
	out = np.zeros((H, W, 3))

	S1 = seg1 > 0
	S2 = seg2 > 0

	overlap = S1 * S2
	out[..., 0][S1 ^ overlap] = 255
	out[..., 2][S2 ^ overlap] = 255
	out[..., 1][overlap] = 255

	return out

def show_kps(img, kps, col=(255, 0, 0), invis_col=(50, 50, 50)):
	"""Return image with all keypoints plotted on"""
	img = tens2rgb(img).copy()

	circle_kwargs = dict(radius=2, thickness=2, lineType=8, shift=0)
	for (x, y, *v) in kps:
		_col = col if v != [0] else invis_col
		img = cv2.circle(img, (int(x), int(y)), color=_col, **circle_kwargs)

	return img

def show_kp_err(img, kp1, kp2, thickness=2):
	"""Overlay keypoints on image, showing lines connecting adjacent keypoints"""
	img = tens2rgb(img).copy()

	img = show_kps(img, kp1, (255, 0, 0))
	img = show_kps(img, kp2, (0, 0, 255))

	for start, end in zip(kp1[..., :2], kp2[..., :2]):
		img = cv2.line(img, tuple(map(int, start)), tuple(map(int, end)), (0, 0, 0), thickness=thickness)

	return img

def produce_grid(entries):
	"""Receives list of lists, containing several possible data types. Converts them all to the correct RGB uint8 format, combines into a single image, and returns.

	Accepted formats:
	Tensor, any device, >= 2 dims (will take first element in all above last 3), >= 3 channels (will take first 3) OR 1 channel (segmentation)
	np.ndarray (same rules as tensor)
	None - fill with blank

	Pads all rows with black images if not enough elements
	"""

	if not isinstance(entries[0], list):
		entries = [entries]  # convert to 2D list of lists

	M = max(map(len, entries))

	H, W = None, None

	rows = []
	for j, raw_row in enumerate(entries):
		row = []
		for i, entry in enumerate(raw_row):
			if entry is None:
				entry = np.zeros((H, W, 3), dtype=np.uint8)

			entry = tens2rgb(entry)

			assert entry.ndim >= 2, f"Arrays for grid must have >= 2 dimensions. Entry ({i}, {j}) has shape {entry.shape}."
			entry = reduce_ax(entry, 3)  # reduce dimensions to just get a single image

			# handle segmentations
			if entry.shape[-1] > 4:  # if last axis is clearly a width/height axis
				entry = seg_to_rgb(reduce_ax(entry, 2))

			entry = entry[..., :3]  # only take first 3 channels

			if i == j == 0:
				H, W, _ = entry.shape

			entry = entry.astype(np.uint8)
			row.append(entry)

		for i in range(M - len(raw_row)):
			row.append(np.zeros((H, W, 3), dtype=np.uint8))  # pad each row with black images if not enough items

		# stack the row images together
		try:
			rows.append(np.hstack(row))
		except:
			raise ValueError(
				f"Could not combine row {j}, of raw shapes: {[x.shape for x in raw_row]}. Attempted conversion to shapes: {[x.shape for x in row]}")

	return np.vstack(rows)


def get_text(string, width, height, backg=(0,0,0), scale=1, linepad=0.2, vertical=False, ):
	"""Get text width backg coloured background, centered to a rectangle of size width, height.
	
	String can be single line, or list of multiple lines"""

	if isinstance(string, str): string = [string]


	font = cv2.FONT_HERSHEY_SIMPLEX
	scale, thick, ltype = scale, 1, 2

	(_, dy), _ = cv2.getTextSize(text=string[0], fontFace=font, fontScale=scale, thickness=thick)
	dy = int(dy * (1 + linepad))

	# Create a zeros image
	if vertical: height, width = width, height
	
	img = np.zeros((height, width, 3), dtype=np.uint8)

	N = len(string)
	for n, line in enumerate(string):
		(w, _), _ = cv2.getTextSize(text=line, fontFace=font, fontScale=scale, thickness=thick)
		y = int(height//2 + dy//2 + dy * (n - 0.5*(N-1)))#+ dy * (0.5 - (N-n)))
		x = width//2 - w//2
		cv2.putText(img, line, (x, y), font, scale, (255, 255, 255), thick, ltype)

	# Rotate the image using cv2.warpAffine()
	if vertical:
		return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	else:
		return img

def put_text(img, string, x, y, width, height, backg=(0,0,0), scale=1, vertical=False):
	"""Place text on an image, with top left corner (x,y), and a given width height.
	White text, black background fixed.
	Vertical flag used to rotate 90 degrees anticlockwise"""

	out = img.copy()
	out[y:y+height, x:x+width] = get_text(string.split('\n'), width, height, scale=scale, backg=backg, vertical=vertical)
	return out


def colourbar(width, height, colours, points=(0, 1), orientation='vertical'):
	"""Produce a colour bar of size width x height.
	At each point in `points`, the colour at point along the horizontal/vertical (depending on `orientation`)
	must be the corresponding colour in `colour`. Between points, linearly interpolate."""

	assert len(colours) == len(points), "Colours to points must be 1-1 correspondence for colourbar"
	colours = np.array(colours)

	img = np.zeros((height, width, 3))
	for (c0, p0, c1, p1) in zip(colours, points, colours[1:], points[1:]):
		if orientation == 'vertical':
			v0, v1 = int(p0*height), int(p1*height)
			img[v0: v1] = c0[None, None, :] + np.linspace(0, 1, v1-v0)[:, None, None] * (c1 - c0)[None, None, :]

		else:
			h0, h1 = int(p0 * width), int(p1 * width)
			img[:, h0:h1] = c0 + np.linspace(0, 1, h1 - h0) * (c1 - c0)

	return img.astype(np.uint8)