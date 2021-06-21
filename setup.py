import requests
from tqdm import tqdm

if __name__ == '__main__':

	try:
		print('Downloading file...')
		
		files = ['https://speed.hetzner.de/100MB.bin',
				'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_1.mp4',
				'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_2.mp4',
				'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_3.mp4',
				'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/video/camera_4.mp4',
				'https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic/raw/master/yolo/yolov4.weights']

		for url in files:

			req = requests.get(url, stream = True)

			name = url.split('/')[-1]
			perc = int(req.headers['content-length']) / 1024
			folder = url.split('/')[-2]

			with open('{}/{}'.format(folder, name), 'wb') as f:
				for data in tqdm(desc = name, iterable = req.iter_content(chunk_size = 1024), total = perc, unit = ' Kb', mininterval = 0.5, maxinterval= 1):
					f.write(data)

		print('Operation completed successfully')

	except Exception as ex:
		
		print(ex)
