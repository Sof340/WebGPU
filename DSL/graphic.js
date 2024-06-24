// Function to create a promise that resolves when the confirm button is clicked
function waitForConfirmation() {
    return new Promise((resolve) => {
        const confirmButton = document.getElementById('confirm-button');

        // Event listener to resolve the promise when the button is clicked
        confirmButton.addEventListener('click', () => {
            resolve();
        }, { once: true });  // { once: true } ensures the event listener is removed after it is triggered
    });
}

// Main function to demonstrate pausing execution
async function main() {
    console.log('Execution paused, waiting for confirmation...');

    // Wait for the confirm button to be clicked
    await waitForConfirmation();

    console.log('Button clicked, resuming execution...');
}

// Call the main function when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    main();
});
