import React, { useState } from 'react';
import axios from 'axios';
import '../App.css'; // Optional: Add styling if needed

const Home = () => {
    const [news, setNews] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setResult(null);
        try {
            const response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/predict`, { news });
            setResult(response.data);
        } catch (error) {
            if (error.response && error.response.data && error.response.data.error) {
                setError(error.response.data.error);
            } else {
                setError('An error occurred. Please try again later.');
            }
        }
    };

    return (
        <div className="container">
            <h1>Fake News Detection</h1>
            <form onSubmit={handleSubmit}>
                <textarea
                    value={news}
                    onChange={(e) => setNews(e.target.value)}
                    rows="6"
                    cols="60"
                    placeholder="Paste the news content here..."
                />
                <button type="submit">Check News</button>
            </form>
            {error && <p className="error">{error}</p>}
            {result && (
                <div className="results">
                    <h3>Results:</h3>
                    <p><strong>Logistic Regression:</strong> {result.logistic_result}</p>
                    <p><strong>Multinomial Naive Bayes:</strong> {result.multi_nb_result}</p>
                </div>
            )}
        </div>
    );
};

export default Home;
