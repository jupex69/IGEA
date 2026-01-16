// Funzione per includere un file HTML dentro un elemento con data-include
function includeHTML() {
    const elements = document.querySelectorAll('[data-include]');
    elements.forEach(el => {
        const file = el.getAttribute('data-include');
        fetch(file)
            .then(response => {
                if (!response.ok) throw new Error('File non trovato: ' + file);
                return response.text();
            })
            .then(data => el.innerHTML = data)
            .catch(err => console.error(err));
    });
}

// Esegui quando la pagina è caricata
document.addEventListener("DOMContentLoaded", includeHTML);