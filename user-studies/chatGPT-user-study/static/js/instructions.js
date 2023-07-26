function add_instructions(placeholder, language){

    var instruction_text = ''
    instruction_text += '<li>In this study, you will be presented with 25 excerpts in English \
        and their respective (human) translations in ' + language+ '. </li>'

    instruction_text += '<li>The goal of this task is to indicate whether or not the two excerpts contain some meaning differences and choose a label that best describes the type of meaning differences (when they exist).</li>'
    if (w_hl){
    instruction_text += '<li>You will also be shown highlighted excerpts. The highlighted excerpts are extracted using an AI system that predicts spans indicative of meaning differences across the two sentences.</li>'
    }

    instruction_text += '<br><strong> When do two excerpts contain some meaning differences? </strong> The excerpts contain some meaning differences, when they convey <strong> mostly the same information, except for some details or nuances </strong> (e.g., some information is added and/or missing on either or both sides; some English words have a more general or specific translation in ' + language + '). You will be further asked to indicate which of the following cases best describes the reason why the excerpts contain some meaning differences: \
    <ul><li>Added (in ' + language + '): The ' + language + ' excerpt contains a piece of information that does not exist in the English excerpt. </li>\
    <li>Added (in English): The English excerpt contains a piece of information that does not exist in the ' + language + ' excerpt. </li>\
    <li>Changed: A piece of information that exists in both the ' + language + ' and the English excerpts does not have the exact same meaning.</ul>'

    $(instruction_text).appendTo(placeholder)
}