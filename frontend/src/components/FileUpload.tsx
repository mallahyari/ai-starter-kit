import React, { useState } from 'react';
import { Upload, message } from 'antd';
import type { UploadProps } from 'antd/es/upload';
import type { UploadFile } from 'antd/es/upload/interface';
import { PlusOutlined } from '@ant-design/icons';

const uploadProps: UploadProps = {
  name: 'file',
  action: 'http://localhost:8000/api/v1/upload',
  beforeUpload(file: File) {
    const allowedTypes = ['application/pdf', 'text/plain'];
    if (!allowedTypes.includes(file.type)) {
      // Show an error message to the user
      message.error(`${file.name} file is not a valid file type.`);
      return Upload.LIST_IGNORE;
    }
    return true;
  },
};

const FileUpload: React.FC<UploadProps> = ({ action }) => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);

  const handleChange: UploadProps['onChange'] = ({
    file: info,
    fileList: newFileList,
  }) => {
    console.log('file', File);

    if (info.status === 'done') {
      message.success(`${info.name} file uploaded successfully`);
    } else if (info.status === 'error') {
      message.error(`${info.name} file upload failed.`);
    }
    setFileList(newFileList);
  };

  return (
    <Upload
      {...uploadProps}
      listType="picture-card"
      fileList={fileList}
      onChange={handleChange}
    >
      <div>
        <PlusOutlined />
        <div style={{ marginTop: 8 }}>Upload</div>
      </div>
    </Upload>
  );
};

export default FileUpload;
