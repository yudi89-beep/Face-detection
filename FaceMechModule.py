import cv2
import mediapipe as mp


class FaceMechDet():
    def __init__(self,staticMode = False, maxFaces=2, minDetCon=0.7,minTrackCon=0.7):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetCon = minDetCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def meshface(self,img,draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:

            for faceLM in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLM, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLM.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    print(id, x, y)
                    face.append([x,y])
                faces.append(face)

        return img,faces



def main():
    cap = cv2.VideoCapture(0)

    detector = FaceMechDet()

    while True:
        success, img = cap.read()
        img,faces = detector.meshface(img)
        #if len(faces) != 0:
            #print(faces[0])

        cv2.imshow('Image', img)
        cv2.waitKey(1)





if __name__ == '__main__':
    main()
