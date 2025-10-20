# Bone Fracture Detection and Classification (Streamlit)

Upload an X-ray image. The app runs YOLOv11 for object detection and EfficientNet-B3 for binary classification: Fractured or Non-Fractured.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

Two-stage X-ray analysis app:
- YOLOv11 object detection (`models/yolov11_trained.pt`)
- EfficientNet-B3 classification (`models/best_model_efficient.pth`) with labels: `Non-Fractured`, `Fractured`

## 🚀 Features

- **AI-Powered Classification**: Uses a custom ResNet deep learning model
- **Two Types Bone Fracture** 
- **Modern UI**: Beautiful, responsive interface with gradient backgrounds
- **Treatment Guidance**: Provides detailed treatment recommendations
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Real-time Prediction**: Fast inference with confidence scores
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## Models
- Detection: YOLOv11 fine-tuned weights at `models/yolov11_trained.pt`.
- Classification: EfficientNet-B3 binary head weights at `models/best_model_efficient.pth`.
- Device: Automatically uses CUDA if available, otherwise CPU.

## Project Structure
- `app.py`: Streamlit UI
- `src/custom_resnet.py`: Inference API (detection + classification)
- `models/`: Place weights `yolov11_trained.pt` and `best_model_efficient.pth`
- `test_img/`: Sample images

## Local Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```
Open the app at `http://localhost:8501`.


## Docker
Build and run:
```bash
docker build -t bone-fracture-app .
docker run --rm -p 8501:8501 -v %cd%/models:/app/bone-fracture-app
```
On Linux/macOS:
```bash
docker run --rm -p 8501:8501 -v $(pwd)/models:/app/bone-fracture-app
```

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only. It is NOT intended for clinical use or medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Medical imaging datasets used for training
- PyTorch and Streamlit communities
- Healthcare professionals who provided guidance
- Open source contributors

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/ruhul-cse-duet/bone-fracture--detection-classification-streamlit/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🔮 Future Enhancements

- [ ] Support for more brain cancer types
- [ ] Integration with medical imaging systems
- [ ] Batch processing capabilities
- [ ] Advanced visualization tools
- [ ] API endpoints for integration
- [ ] Mobile app development
- [ ] Multi-language support

---
## Author
[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/);  
Email: ruhul.cse.duet@gmail.com

**Disclaimer**: This application is for educational and research purposes only. It should not be used for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals for medical advice.

**Made with ❤️ for the medical AI community**

