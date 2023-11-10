import React from 'react';
import { Avatar, Typography, Space, Flex } from 'antd';
import './styles.css';

const { Text } = Typography;

interface MessageProps {
  text: string;
  isUser: boolean;
}

const Message: React.FC<MessageProps> = ({ text, isUser }) => {
  return (
    <Flex vertical={false} gap={3} justify="flex-start">
      {isUser ? (
        <Avatar
          size={40}
          gap={4}
          shape="square"
          style={{ fontSize: '16px', minWidth: '40px' }}
        >
          User
        </Avatar>
      ) : (
        <Avatar
          size={40}
          gap={4}
          shape="square"
          style={{
            backgroundColor: '#f56a00',
            fontSize: '16px',
            minWidth: '40px',
          }}
        >
          AI
        </Avatar>
      )}

      <Space
        direction="vertical"
        size={4}
        style={{ marginLeft: 12, textAlign: 'left' }}
      >
        <Text className="message-text">{text}</Text>
      </Space>
    </Flex>
  );
};

export default Message;
