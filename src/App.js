import React, { useState, useRef } from 'react';
import { getFunctions, httpsCallable } from 'firebase/functions';
import MatrixRain from './matrixChatPage';

const MatrixChatPage = () => {
  const [terminalState, setTerminalState] = useState('closed');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);



  const handleSubmit = async (e) => {
    e.preventDefault();
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: 'User' }]);
      setInput('');
      setTerminalState('open');
      setIsLoading(true);

      try {
        const functions = getFunctions();
        const chatFunction = httpsCallable(functions, 'chat');
        console.log('Calling Firebase function with input:', input);
        const result = await chatFunction({ message: input });
        console.log('Firebase function response:', result);
        if (result && result.data) {
          setMessages(prev => [...prev, { text: result.data.result, sender: 'System' }]);
        } else {
          throw new Error("Unexpected response from Firebase function");
        }
      } catch (error) {
        console.error("Error calling Firebase function:", error);
        console.error("Error details:", error.code, error.message, error.details);
        setMessages(prev => [...prev, { text: `Error: ${error.message}. Code: ${error.code}. Details: ${JSON.stringify(error.details)}`, sender: 'System' }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const inputStyle = {
    width: '100%',
    padding: '10px',
    backgroundColor: 'transparent',
    border: 'none',
    borderBottom: '1px solid #00ff00',
    color: '#00ff00',
    fontFamily: 'monospace',
    fontSize: '16px',
    outline: 'none',
  };

  const buttonStyle = {
    padding: '10px 20px',
    backgroundColor: 'transparent',
    color: '#00ff00',
    border: '1px solid #00ff00',
    cursor: 'pointer',
    fontFamily: 'monospace',
    fontSize: '16px',
    transition: 'background-color 0.3s, color 0.3s',
  };

  const renderTerminal = () => {
    const terminalStyle = {
      position: 'fixed',
      left: 0,
      width: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.9)',
      borderTop: '2px solid #00ff00',
      transition: 'all 0.3s ease-in-out',
      zIndex: 1000,
    };

    if (terminalState === 'closed') {
      return null;
    }

    if (terminalState === 'minimized') {
      return (
        <div
          style={{
            ...terminalStyle,
            bottom: 0,
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 20px',
            cursor: 'pointer',
          }}
          onClick={() => setTerminalState('open')}
        >
          <span>Chat Terminal (Click to expand)</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setTerminalState('closed');
            }}
            style={{
              backgroundColor: 'transparent',
              border: 'none',
              color: '#00ff00',
              fontSize: '20px',
              cursor: 'pointer',
            }}
          >
            ×
          </button>
        </div>
      );
    }

    return (
      <div style={{
        ...terminalStyle,
        bottom: 0,
        height: '50%',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px',
          borderBottom: '1px solid #00ff00'
        }}>
          <span>Terminal</span>
          <div>
            <button onClick={() => setTerminalState('minimized')} style={{
              backgroundColor: 'transparent',
              border: 'none',
              color: '#00ff00',
              marginRight: '10px',
              cursor: 'pointer'
            }}>
              _
            </button>
            <button onClick={() => setTerminalState('closed')} style={{
              backgroundColor: 'transparent',
              border: 'none',
              color: '#00ff00',
              fontSize: '20px',
              cursor: 'pointer'
            }}>
              ×
            </button>
          </div>
        </div>
        <div style={{
          flexGrow: 1,
          overflowY: 'auto',
          padding: '20px'
        }}>
          {messages.map((msg, index) => (
            <div key={index} style={{ marginBottom: '10px' }}>
              <strong>{msg.sender}:</strong> {msg.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} style={{
          padding: '10px',
          borderTop: '1px solid #00ff00',
          display: 'flex',
          alignItems: 'center',
        }}>
          <span style={{ color: '#00ff00', marginRight: '10px' }}>{'>'}</span>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter command..."
            style={inputStyle}
          />
        </form>
      </div>
    );
  };

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      color: '#00ff00',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'monospace'
    }}>
      <MatrixRain />
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: '20px',
        flexGrow: 1,
        position: 'relative',
        zIndex: 1,
      }}>
        <h1 style={{ textAlign: 'center', marginBottom: '20px' }}>Welcome to the Matrix</h1>
        <p style={{ textAlign: 'center', marginBottom: '20px' }}>Enter the digital realm and uncover the truth.</p>
        <form onSubmit={handleSubmit} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <div style={{ position: 'relative', width: '70%' }}>
            <span style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', color: '#00ff00' }}>{'>'}</span>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              style={{
                ...inputStyle,
                paddingLeft: '20px',
              }}
              disabled={isLoading}
            />
          </div>
          <button type="submit" style={{ ...buttonStyle, marginLeft: '10px' }} disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Submit'}
          </button>
        </form>
      </div>
      {renderTerminal()}
    </div>
  );
};

export default MatrixChatPage;