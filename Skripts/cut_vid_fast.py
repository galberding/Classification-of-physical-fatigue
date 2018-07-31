import os
import glob
import cv2

# cap = cv2.VideoCapture("/vol/prt/analysis/heartrate_au/video/front/00000.mp4")


# 1s = 1000ms


class VideoHandler():
    """Cutting videos on specific timestamps.
        name: name of the file
        search dir: parent directory where the search starts
        The file is located while initializing.
        Exception if Video is not found.
    """
    def __init__(self, file_name, search_dir, out_dir):
        self.file_name = file_name
        self.search_dir = search_dir
        self.out_dir = out_dir
        self.file_path = None
        self.snippet_paths = []
        # self.fps = self.get_fps()
        self.fps = 0
        print("start search")
        for path, subdir, files in os.walk(search_dir):
            # print("{} {} {}".format(path, subdir, files))
            for file in files:
                # print(file)
                if glob.fnmatch.fnmatch(file,self.file_name):
                    self.file_path = os.path.join(path, file)
                    break
        if self.file_path:
            print("{} found at {}".format(self.file_name, self.file_path))
        else:
            print("No video!")

    def get_fps(self):
        if not self.file_path:
            print("No file found!")
            return 0.0
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if self.fps == 0:
            print("bad Frame rate")

        cap.release()
        return fps

    def cut_vid(self, start_stamp, maxes):
        """ Create snippets of interesting spots of a given video.
        
            Parameters:
            -----------
            
            start_stamp: timestamp with wihich the video starts.
            
            maxes: list of tuples with timestamp and max value
            
            Return: 
            -------
            None if something fails
            Path to snip otherwise
        """

       
        if (not self.file_path):
            print("{} not found".format(self.file_name))
            return

        if not maxes:
            print("{} has no maxes".format(self.file_name))
            return

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        cap = cv2.VideoCapture(self.file_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            print("Unable to open {}".format(self.file_name))
            return

        # Change directory and create new for the current samples
        os.chdir(self.out_dir)
        os.makedirs(self.file_name[:-4], exist_ok=True)
        os.chdir(self.file_name[:-4])

        ret1, frame1 = cap.read()
        width = frame1.shape[1]
        heigth = frame1.shape[0]
        stamp_times = {}
        writer_states = {}
        time_range = 25*1000
        count = 0
        init_sample = True
        for stamp, rate in maxes:

            spot = (stamp - start_stamp)*1000
            print("start: {} current {} Spot {} in {}".format(start_stamp, stamp, spot, self.file_name))
            if spot < 0:
                # TODO: Inform if that happens!
                fo = open("probles.csv", "w+")
                fo.write("{},{},{},\n".format(self.file_name,start_stamp, stamp))
                fo.close()
                continue
            # out_file_name = self.out_dir + self.file_name[:-4]
            out_file_name =  self.file_name[:-4]
            out_file_name += "_snip_{:02}_{}_{}.avi".format(count, stamp, rate)
            self.snippet_paths.append(os.path.join(os.getcwd(),out_file_name))
            stamp_times[spot] =  cv2.VideoWriter(out_file_name, fourcc, self.fps, (width, heigth))
            writer_states[spot] = 0
            count += 1

        # cap.set(cv2.CAP_PROP_POS_MSEC, 0)

        writer = len(stamp_times)
        print(writer)
        counter = 1
        for time, output_cap in stamp_times.items():
            start = time + 5000
            end = time + time_range +5000
            cap.set(cv2.CAP_PROP_POS_MSEC, start)
            print("Progress: {} / {}".format(counter, writer))
            counter += 1
            while end >= cap.get(cv2.CAP_PROP_POS_MSEC):
                ret, frame = cap.read()
                if not ret:
                    break
                output_cap.write(frame)



        for time, output_cap in stamp_times.items():
            output_cap.release()

        cap.release()

        return self.snippet_paths

#
# if __name__ == '__main__':
#     import environment_techfak as paths
#     # from cut_vid_fast import VideoHandler
#     vh = VideoHandler("00000.mp4", paths.VIDEO_SEARCH_DIR, paths.VIDEO_TEST_SNIP_SAFE_DIR)
