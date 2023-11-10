import React, { useState, useEffect, useRef } from 'react';
import { Input, List, Button, Flex, Row, Col } from 'antd';
import clsx from 'clsx';
import Message from './Message';
import './styles.css';

const { TextArea } = Input;

interface Message {
  text: string;
  isUser: boolean;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [streamedData, setStreamedData] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    if (inputValue.trim() === '') return;
    setMessages([...messages, { text: inputValue, isUser: true }]);
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: 'Loading...', isUser: false },
    ]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: inputValue }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response from the server.');
      }

      // Process the streaming response
      const reader = response.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder('utf-8');

        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          const text = decoder.decode(value);
          setStreamedData((prevData) => prevData + text);
        }
      }
    } catch (error) {
      console.error('Error fetching and streaming response:', error);
    } finally {
      setStreamedData('');
      setLoading(false);
    }
  };

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    if (streamedData !== '') {
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1), // Remove the loading message
        { text: streamedData, isUser: false },
      ]);
    }
  }, [streamedData]);

  return (
    <Flex gap="middle" vertical style={{ padding: '24px' }}>
      <List
        itemLayout="horizontal"
        size="large"
        dataSource={messages}
        renderItem={(item, index) => (
          <List.Item
            className={clsx(item.isUser ? 'user-message' : 'ai-message')}
          >
            <Message text={item.text} isUser={item.isUser} />
          </List.Item>
        )}
      />
      <div ref={chatEndRef} />
      <Row gutter={0}>
        <Col flex="auto">
          <div
            style={{
              width: '100%',
              margin: 'auto',
              boxShadow: '0 0px 14px rgba(0, 0, 0, 0.1)',
              borderRadius: '5px',
            }}
          >
            <TextArea
              className="no-outline"
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here..."
              autoSize={{ minRows: 1, maxRows: 6 }}
              style={{
                minHeight: '60px',
                maxHeight: '120px',
                boxShadow: '0 0px 14px rgba(0, 0, 0, 0)',
                boxSizing: 'border-box',
                paddingLeft: '10px',
                marginTop: '20px',
                marginBottom: '20px',
                paddingRight: '80px',
                position: 'relative',
                border: 'none',
                outline: 'none !important',
                resize: 'none',
              }}
            />
            <Button
              type="primary"
              onClick={handleSubmit}
              loading={loading}
              style={{
                position: 'absolute',
                top: '50%',
                transform: 'translateY(-50%)',
                right: '20px',
              }}
            >
              Send
            </Button>
          </div>
        </Col>
      </Row>
    </Flex>
  );
};

export default Chat;
