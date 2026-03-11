from ultralytics import YOLO

if __name__ == '__main__':
    
    # โหลดโมเดล
    model = YOLO("yolo26n.pt") 
    
    # เริ่มเทรน
    results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, device=0)