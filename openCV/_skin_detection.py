import cv2
import numpy as np

def unique(a):
    """ remove duplicate columns and rows
        from http://stackoverflow.com/questions/8560440 """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

def skin_detection(image, haarcascades, flood_diff=3, min_face_size=(30,30),
        num_iter=3, verbose=False, step=1):
    faces = haarcascades.detectMultiScale(image, minSize=min_face_size)
    if len(faces) == 0:
        raise Exception('no faces')
    else:
        print(f"detect {len(face)} face(s) ")

    image_original = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image2 = np.copy(image_original)
    image2[image2==[0,0,0]] = 1

    skin_color = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        image_face = image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
        arr = [image_face[image_face.shape[0]//2, image_face.shape[1]//2]]
        skin_color[i] = np.array(arr)

    mask = np.zeros(image_original.shape)
    for i in range(num_iter):
        # for each pixel, call floodFill(image2) if it is close to skin_color
        for y in range(0, image2.shape[0], step):
            if verbose:
                print ('iter: %d, y:%d' % (i, y))
            for x in range(0, image2.shape[1], step):
                color = image2[y,x]
                if (color!=(0,0,0)).any():
                    if any((np.abs(skin_color-color)<=(flood_diff,)*3).all(1)):
                        cv2.floodFill(image2, None, (x,y), (0, 0, 0),
                                loDiff=(flood_diff,)*3, upDiff=(flood_diff,)*3)

        # update mask image and skin_color
        mask[image2==(0,0,0)] = 255
        skin_color = image_original[mask.nonzero()]
        skin_color.shape = (skin_color.shape[0]//3, 3)
        skin_color = unique(skin_color)

    mask = np.bool_(mask)
    return mask

def test():
    filename = './inputs/test.jpg'
    filename_out = (''.join(filename.split('.')[:-1])
                    + '_out%d.' + filename.split('.')[-1])
    image = cv2.imread(filename)
    hc = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    # hc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    mask = skin_detection(image, hc, verbose=True)
    cv2.imwrite(filename_out % 1, image)
    cv2.imwrite(filename_out % 2, np.uint8(mask)*255)
    image[mask==False] = 0
    cv2.imwrite(filename_out % 3, image)

if __name__ == '__main__':
    test()
