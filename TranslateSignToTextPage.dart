import 'package:flutter/material.dart';
// import 'package:image_picker/image_picker.dart';
import 'LoginPage.dart';
import 'TranslateTextToSignPage.dart';
import 'ChatWithGPTPage.dart';
import 'camera_page.dart';
import 'HomePage.dart';

class TranslateSignToTextPage extends StatelessWidget {
  const TranslateSignToTextPage({Key? key}) : super(key: key);
  void _openCamera(BuildContext context) async {
    //   final picker = ImagePicker();
    //   final pickedImage = await picker.pickVideo(source: ImageSource.camera);
    //
    //   if (pickedImage != null) {
    print('in function');
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => CameraPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Translate Sign to Text'),
        actions:[
          IconButton(
            icon: Text(
              '☰',
              style: TextStyle(
                fontSize: 24,
                color: Colors.white,
              ),
            ),
            onPressed: () {
              // Show a menu or perform another action
              final RenderBox overlay =
              Overlay.of(context).context.findRenderObject() as RenderBox;
              showMenu(
                context: context,
                position: RelativeRect.fromLTRB(
                  overlay.size.width - 100, // Adjust the X position
                  56.0, // This should match the height of your AppBar
                  0.0,
                  0.0,
                ),
                items: [
                  PopupMenuItem(
                    child: Text('Sign Out'),
                    value: 'signOut',
                  ),
                  PopupMenuItem(
                    child: Text('Translate Text to Sign'),
                    value: 'textToSign',
                  ),
                  PopupMenuItem(
                    child: Text('Chatbot'),
                    value: 'chat',
                  ),
                  PopupMenuItem(
                    child: Text('Home'),
                    value: 'home_page',
                  ),
                ],
              ).then((value) {
                if (value != null) {
                  switch (value) {
                    case 'signOut':
                      Navigator.pushReplacement(context,
                          MaterialPageRoute(builder: (context) => LoginPage()));
                      break;
                    case 'textToSign':
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => TranslateTextToSignPage()));
                      break;
                    case 'chat':
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => ChatWithGPTPage()));
                      break;
                    case 'home_page':
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => HomePage()));
                      break;
                    default:
                  }
                }
              });
            },
          ),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => _openCamera(context),
              child: Text(
                'Open Camera',
                style: TextStyle(fontSize: 18, color: Colors.white),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).hintColor,
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
