console.log("✅ main.js loaded");

$(document).ready(function () {

    //eel.init()

    $('.tlt').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "bounceIn",
        },
        out: {
            effect: "bounceOut",
        },
    });

    // Siri configuration
    const siriWave = new SiriWave({
        container: document.getElementById("siri-container"),
        width: 800,
        height: 200,
        style: "ios9",
        amplitude: 1,
        speed: 0.3,
        autostart: true
    });

    // Siri message animation
    // $('.siri-message').textillate({
    //     loop: true,
    //     sync: true,
    //     // in: {
    //     //     effect: "fadeInUp",
    //     //     sync: true,
    //     // },
    //     // out: {
    //     //     effect: "fadeOutUp",
    //     //     sync: true,
    //     // },

    // });

    //mic button event
    $("#MicBtn").click(function () {
        eel.MicSound();
        $("#Oval").attr("hidden", true);
        $("#SiriWave").attr("hidden", false);
    
        // Start the Python-driven loop
        eel.startSession()(function() {
          // Once Python returns, show the hood again
          $("#Oval").attr("hidden", false);
          $("#SiriWave").attr("hidden", true);
        });
      });
});
