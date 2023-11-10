import React from 'react';
import './App.css';
import ChatComponent from './components/ChatComponent';
import { Flex, Typography, Upload } from 'antd';
import FileUpload from './components/FileUpload';

const { Title, Paragraph } = Typography;

function App() {
  const beforeUpload = (file: File) => {
    // Check if the file is valid
    const allowedTypes = ['application/pdf', 'text/plain'];
    if (!allowedTypes.includes(file.type)) {
      // Show an error message to the user
      return false;
    }
    return true;
  };

  const onFileChange = (fileList: File[]) => {
    // Do something with the uploaded files
  };
  return (
    <div className="App">
      <Flex vertical>
        <div style={{ margin: 'auto', textAlign: 'left' }}>
          <Title> Chat with Your Documents Locally</Title>
          <Paragraph style={{ fontSize: '18px' }}>
            You can upload PDF files and start asking questions about them.
          </Paragraph>
        </div>
        <FileUpload />
        <ChatComponent />
      </Flex>
    </div>
  );
}

export default App;
