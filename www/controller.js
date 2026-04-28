$(document).ready(function () {
    // Display Speak Message
    eel.expose(DisplayMessage);
    function DisplayMessage(msg) {
        $(".siri-message li:first").text(msg);
        $('.siri-message').textillate('start');

        // Also append to log if it's not the initial system prompt
        if (!msg.startsWith("You are a virtual patient")) {
            const role = msg === "Goodbye doctor!" ? "Patient" : $(".siri-message li:first").text() === msg ? "Patient" : "Doctor";
            $("#conversation-log").append(`<p><strong>${role}:</strong> ${msg}</p>`);
            const log = $("#conversation-log").get(0);
            log.scrollTop = log.scrollHeight;
        }
    }

    // Display hood
    eel.expose(ShowHood);
    function ShowHood() {
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
    }

});