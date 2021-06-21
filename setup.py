import requests
import warnings
from tqdm import tqdm

if __name__ == '__main__':

	# warnings.filterwarnings('ignore')

	files = ['https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_1.mp4',
		     'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_2.mp4',
		     'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_3.mp4',
		     'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_4.mp4',
		     'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/yolo/yolov4.weights']

	for url in files:

		req = requests.get(url, stream = True)

		filename = url.split('/')[-1]
		folder = url.split('/')[-2]

		with open('{}/{}'.format(folder, filename), 'wb') as f:
			for data in tqdm(iterable = req.iter_content(chunk_size = 1024), total = int(req.headers['content-length']) / 1024, unit = ' Kb', desc = (filename.replace('_', ' ')).capitalize()):
				f.write(data)

	print("Done!")
