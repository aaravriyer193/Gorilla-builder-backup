import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// 1. Middleware
app.use(cors()); // Allow cross-origin requests (Vital for WebContainer)
app.use(express.json()); // Parse JSON bodies

// 2. Test Route (Root)
app.get('/', (req, res) => {
  res.send(`
    <div style="font-family: sans-serif; text-align: center; padding: 20px;">
      <h1>🚀 Server is Running!</h1>
      <p>Your Node.js environment is fully active.</p>
      <p>Try the API: <a href="/api/hello">/api/hello</a></p>
    </div>
  `);
});

// 3. API Route Example
app.get('/api/hello', (req, res) => {
  res.json({ 
    status: 'success',
    message: 'Hello from the backend!',
    timestamp: new Date().toISOString()
  });
});

// 4. Start Server
app.listen(port, () => {
  console.log(`⚡ Server running at http://localhost:${port}`);
});