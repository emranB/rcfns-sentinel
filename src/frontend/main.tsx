import { createRoot } from 'react-dom/client';
import App from './App';
import './chat.css';

const root = document.getElementById('root');
if (root) createRoot(root).render(<App />);
