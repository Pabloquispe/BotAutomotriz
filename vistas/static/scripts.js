let chatInitialized = false; // Estado para controlar si el chat ha sido inicializado

function toggleChat() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.classList.toggle('hidden');
    
    if (!chatInitialized) {
        fetch('/api/welcome')
            .then(response => response.json())
            .then(data => {
                appendMessage('bot', data.message);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        chatInitialized = true; // Marcar el chat como inicializado para evitar múltiples llamadas
    }
}

function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    appendMessage('user', userInput);
    document.getElementById('user-input').value = '';

    fetch('/conversacion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        appendMessage('bot', data.message);
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage('bot', 'Hubo un error al procesar tu mensaje. Por favor, inténtalo de nuevo.');
    });
}

function appendMessage(sender, message) {
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
    messageElement.textContent = message;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function checkEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('bot-icon').addEventListener('click', toggleChat);
});
