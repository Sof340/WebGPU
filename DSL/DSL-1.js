class WebGPUManager {
    /**
     * @method initialization : Sets up the adapter and device, returns an error message in case of failure.
     */
    async initialization() {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            let errorMessage = `An error occured while trying to procure the adapter please check the following points :
            Your internet browser is compatible with WebGPU(you can do so by visiting this website: https://caniuse.com/webgpu .
            You GPU is fonctionnel (no hardware malfunctions).
            Your GPU is free and is not in use in other tabs, browser or softwares.`;
            document.write(errorMessage);
            throw new Error(errorMessage);
        }
        const device = await adapter.requestDevice();
        this.device = device;// the GPU device that will runs our WGSL code.
    }


    /**
     * contructor represents a method that runs when a new intance of the class is created.
     */
    constructor() {
        this.shader_code = '';// a string that will hold the translated function (WGSL code).
        this.functionName = ''; // A variable that contains the name of the function that is getting translatted.
        this.mainFunctionName = ''; // A variable that contains the name of the main function.
        this.shaderCodeHead = ''; // this variable will store the string related to the input buffers (in WGSL).


        this.inputBuffers = {};// an Object that stores the names and refferences of the input data (variables).
        this.outputBuffers = {};// an Object that stores the names and refferences of the output data (variables).
        this.linkedVariables = {};// an Object that stores the link between the input and output variables (from which input buffer will the result buffer copy data).
        this.functionVariables = {};// an Object that stores the name of a function and it's variables.
        this.variablesOfFunction = {};// This Object stores the name : type of the arguments and variables of the function to translate.
        this.inputVariablesDict = {};//this Object stores the parameters of the function and their type (in WGSL).
        this.theTranscriptOfFunctions = {};// this Object contains the function name as key and an array of arrays as values ,
        //where each array corresponds too a line of the function, it is used for when the user introduces a change in the types of variables(via graphic interface).
        this.dictFunctionOutputTypes = {}; // a Object that stores the output type of all the functions, needed in case the types are changed.

        this.counter = 0; // a counter for the bindings of the same group.
        this.workgroups_size = 1; // a variable that contains the value of the workgroup_size

        this.functionTextAsArray = [];// this arrays contains each line of the functions to translate as an arrays, we use this variable to store the function texte in array format in the Object defined above.

        this.variableTypesChanged = false; // this boolean variable tells us if the use changed any variables type using the graphic interface.

        this.redirectConsoleLog(); // Call the redirection function
    }


    /**
     * @method createInputBuffer: represent a method used to copy data into the GPU for later use
     * @param {string} name : a string that represents the name the variable will have in the WGSL code.
     * @param {same as the input variable} data : the input data to be copied into the GPU.
     * @param {number} group_binding : identifies the binding group which the input variable belongs to, default value is 0.
     */
    createInputBuffer(name, data, group_binding = 0) {
        const dataSize = data.byteLength;
        const buffer = this.device.createBuffer({
            size: dataSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false
        });
        // We write the data into the created buffer.
        this.device.queue.writeBuffer(buffer, 0, data);

        this.inputBuffers[name] = buffer; // fill ou the Object { name: buffer refference }.

        const J_typeOfArray = data.constructor.name;
        let typeOfArray = '';
        switch (J_typeOfArray) {
            case "Int32Array":
                typeOfArray = 'array<i32>';
                break;
            case "Float32Array":
                typeOfArray = 'array<f32>';
                break;
            case "Uint32Array":
                typeOfArray = 'array<u32>';
                break;

        }
        this.shaderCodeHead += '@group(' + String(group_binding) + ') ' + '@binding(' + String(this.counter) + ') var<storage, read_write> ' + name + ' : ' + String(typeOfArray) + ';\n';
        this.counter++;
    }

    /**
     * @method createOutputBuffer: represents a method to get results from the GPU after computation.
     * @param {string} name : a string that represents the name the variable will have in the WGSL code.
     * @param {string} linked_variable : a string that holds the name of the variable (associated with a buffer) to link the ouput variable to.
     * @param {same as the input variable} data : the input data to be copied into the GPU.
     */
    createOutputBuffer(name, linked_variable, data) {
        const dataSize = data.byteLength;
        const buffer = this.device.createBuffer({
            size: dataSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.outputBuffers[name] = buffer;// fill ou the Object { name: buffer refference }.
        this.linkedVariables[name] = linked_variable;// fill ou the Object { name: linked_variable }.
    }

    /**
   * @method processFunction : a method that pre-process the input function then pass it into the kernel for convertion.
   * @param {function} func : the call of the javascript function to be translated into WGSL. 
   * @param {string} outputType : a string representing the type of the output of the function.
   * @param  {...any} args : the parameters of the function, inputed as in a normall call to this function.
   */
    processFunction(func, outputType, ...args) {

        // Here we fill the inputVariablesDict with the javascript type of each parameter.
        for (let x = 0; x < args.length; x++) {
            this.inputVariablesDict[args[x]] = typeof (args[x]);
        }
        //console.log(this.inputVariablesDict);// just to verify the correct output

        // Now we need to translate the type of each arguments into the corresponding type in WGSL (this needs more thinking)
        Object.keys(this.inputVariablesDict).forEach(key => {
            if (this.inputVariablesDict[key] == 'number') {
                this.inputVariablesDict[key] = 'f32';
            } else if (this.inputVariablesDict[key] == 'object') {
                let tempArray = key;
                let arraySize = 0;
                for (let x = 0; x < key.length; x++) {
                    if (tempArray[x] == ',') {
                        arraySize++;
                    }
                }
                arraySize++;
                this.inputVariablesDict[key] = 'array<f32,' + String(arraySize) + '>';
            }
        });

        // Simple printing.
        /*Object.keys(this.inputVariablesDict).forEach(key => {
            console.log(this.inputVariablesDict[key]);
        });*/


        const funcString = func.toString();// Convert the function to a string.
        const lines = funcString.split("\n");// cut the resulted string into lines.
        const delimiters = /[\s]/;// we prepare the specific delimiters to use to clean the lines (here we have : spaces and ',' and ';').

        if (this.shader_code.length == 0) {
            this.shader_code = this.shaderCodeHead; // we input the head of the wgsl code into the shader_code variable that will contain the reste of the code(plus the translated functions).
        }

        for (let x = 0; x < lines.length; x++) {
            let word = lines[x].split(delimiters);// we clean the selected line with the chosen delemitters.

            let newArr = [];

            // this part is just to separate the '(' and ')' from the arguments 
            word.forEach(entry => {
                let parts = entry.split(/(\(|\)|\[|\]|\,|\.)/); // Split where '(' is preceded by a space or at the start of the string
                // Add each part to the new array
                parts.forEach(part => {
                    newArr.push(part.trim()); // Trim to remove any leading or trailing whitespace
                });
            });



            word = newArr.filter(entry => entry);// we filters our array to take out all the empty entries

            this.functionTextAsArray.push(word);
            //shader_code += word.join(' ') + '\n';
            this.shader_code += this.kernel(word, x, outputType) + '\n';// a call to the kernel procedure that translates the code line.

        }

        this.variablesOfFunction["return"] = outputType;
        this.functionVariables[this.functionName] = this.variablesOfFunction;// we save the information about each function here.
        this.theTranscriptOfFunctions[this.functionName] = this.functionTextAsArray;
        this.inputVariablesDict = {};// we empty this Object for future uses.
        this.variablesOfFunction = {};// we empty this Object for future uses.
        this.functionTextAsArray = [];// we empty the array for future uses.


        // output the resulted translated function.
        console.log(this.shader_code);

    }


    /**
 * @method kernel : an internal method that converts string lines in javascript into WGSL. 
 * @param {Array} line : an array containing a formmatted line of the javascript function.
 * @param {number} x : the number of the current line.
 * @param {string} outputType : a string depecting the type of the output of the function
 * @returns {string} lineOfCode : the translated line of code (Written in WGSL)
 * */
    kernel(line, x, outputType) {
        let lineOfCode = '';

        //Here we transform the first line of the function (the function declaration).
        if (x == 0) {
            line = line.filter(item => item !== ',')
            const values = Object.values(this.inputVariablesDict);
            this.functionName = line[1]; // we stores the function name for later.
            lineOfCode += 'fn ' + line[1] + line[2];

            this.dictFunctionOutputTypes[line[1]] = outputType; // we fill the Object with functionName : type of the return .

            let index = 0;
            for (let word = 3; word < line.length - 2; word++) {
                if (line[word + 1] == ')') {
                    lineOfCode += line[word] + ': ' + values[index];
                    this.variablesOfFunction[line[word]] = values[index];
                } else {
                    lineOfCode += line[word] + ': ' + values[index] + ', ';
                    this.variablesOfFunction[line[word]] = values[index];
                    index += 1;
                }
            }
            lineOfCode += ')' + ' -> ' + outputType + ' ' + line[line.length - 1];


            lineOfCode += '\n';
            return lineOfCode;
        }

        //If the array contains only one element it means it's a closing bracket '{'.
        else if (line.length == 1) {
            if (line[0] == '{') lineOfCode += '{' + '\n';
            else if (line[0] == '}') lineOfCode += '}' + '\n';
            else lineOfCode += line[0] + '\n';

            return lineOfCode;
        }

        else {
            for (let j = 0; j < line.length; j++) {
                switch (line[j])
                // replace 'let','var','const' with their equivlant in WGSL
                {
                    case 'let':
                        this.variablesOfFunction[line[j + 1]] = "f32";
                        line[j] = 'var';
                        break;
                    case 'var':
                        this.variablesOfFunction[line[j + 1]] = "f32";
                        line[j] = 'var';
                        break;
                    case 'const':
                        this.variablesOfFunction[line[j + 1]] = "f32";
                        line[j] = 'let';
                        break;
                    // replace return with its equivlant
                    case 'return':
                        line[j] = 'return';
                        break;
                    // replace JS array declaration to WGSL declaration
                    case '[':
                        if (line[j - 1] == '=') // dectect if the line is a declaration of an array
                        {
                            let elements = ''; // to store all array elements
                            let i = j + 1;
                            let num_vir = 0;
                            while (i < line.length) {
                                if (line[i] == ',') {
                                    i++;  //skip the ',' charecter
                                    num_vir++;
                                }
                                else if (line[i] == ']') {
                                    break; // stop when finding ']' charecter (array ends)
                                }
                                else {

                                    elements += line[i] + ',';// count the number of elements 
                                    i++;
                                }

                            }
                            num_vir++;
                            this.variablesOfFunction[line[j - 2]] = "array<f32," + String(num_vir) + ">";

                            elements = elements.slice(0, -1); // remove the last charecter because it is always a ',' look above
                            lineOfCode += 'array(' + elements + ')';
                            return lineOfCode;
                        }
                    case 'switch':
                        line = line.filter(item => item !== ')');
                        line = line.filter(item => item !== '(');
                    case 'Math':
                        line = line.filter(item => item !== 'Math');
                        line[line.indexOf('Math') + 1] = '';// remove the '.' that comes after math module call
                        break;
                    case "\'use":
                        return '';
                }
                if (line[j] == '.' || line[j + 1] == '.') {
                    lineOfCode += line[j];
                } else {
                    lineOfCode += line[j] + ' ';
                }

            }


            lineOfCode += "\n";
            return lineOfCode;

        }



    }

    /**
     * @method createPipeline: creates the compute pipeline that is to be executed.
     * @param {string} description: a simple description of the pipeline usefull for debugging(appears in error messages). 
     */
    createPipeline(description) {

        const module = this.device.createShaderModule({
            label: description + ' module',
            code: this.shader_code,
        });

        this.pipelineLayout = this.device.createPipelineLayout({
            label: description + "Layout",
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.pipeline = this.device.createComputePipeline({
            label: description,
            layout: this.pipelineLayout,
            compute: {
                module,
                entryPoint: this.mainFunctionName,
            },
        });
    }

    /**
     * @method createBindGroup : a method that generates a bindGroup automatically when created 
     * @param {string} description : a short string referencing the bindgroup to be created, very usefull as a flag in case of errors.
     */
    createBindGroup(description) {

        const bindings = [];// An array that will stores the format of our bindings 
        const bindingsLayout = [];

        let i = 0;

        // We fill up the bindings array.
        for (const key in this.inputBuffers) {
            if (this.inputBuffers.hasOwnProperty(key)) {
                bindings.push({
                    binding: i,
                    resource: {
                        buffer: this.inputBuffers[key]
                    }
                });

                bindingsLayout.push({
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }

                });
                i++;
            }
        }



        console.log(bindingsLayout);
        console.log(bindings);
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: "Layout" + description,
            entries: bindingsLayout
        });

        // Creations of the binding group
        this.bindGroup = this.device.createBindGroup({
            label: description,
            layout: this.bindGroupLayout,// asks for a simple bindGroup Layout since only compute shaders are used;
            entries: bindings,
        });

    }

    /**
     * @method createEncoder : Sets up the commands that are to be executed by the GPU upon call.
     * @param {string} description : a short label depecting our encoder.
     * @param {vec3} workGroupsSize : a 3D arrays that defines the number of independents threads to be executed.
     */
    async createEncoder(description, workGroupsSize) {
        const encoder = this.device.createCommandEncoder({
            label: description,
        });
        const pass = encoder.beginComputePass({
            label: description + 'pass',
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(workGroupsSize);
        pass.end();

        for (let key in this.linkedVariables) {
            const inputBuffer = this.inputBuffers[this.linkedVariables[key]];
            const outputBuffer = this.outputBuffers[key];
            encoder.copyBufferToBuffer(inputBuffer, 0, outputBuffer, 0, inputBuffer.size);
        }


        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);


    }

    /**
     * @method getResult: used to retrieve the results of the computation.
     */
    async getResult() {
        for (const key in this.outputBuffers) {
            await this.outputBuffers[key].mapAsync(GPUMapMode.READ);
            ////////////////////////////////////
            const result = new Float32Array(this.outputBuffers[key].getMappedRange().slice());
            ///////////////////////////////////
            this.outputBuffers[key].unmap();
            console.log('result', result);
        }
    }






    /**
     * @method kernelUpdate : generates an updated shader code based on the corrections inputted.
     * @param {Array} line : an array containing a formmatted line of the javascript function.
     * @param {number} x : the number of the current line.
     * @param {string} outputType : a string depecting the type of the output of the function.
     * @param {Object} variablesType: an Object that stores the type of each variable of each funtion.
     * @returns {string} lineOfCode : the translated line of code (Written in WGSL)
     * */
    kernelUpdate(line, x, outputType, variablesType) {
        let lineOfCode = '';

        //Here we transform the first line of the function (the function declaration).
        if (x == 0) {
            line = line.filter(item => item !== ',')
            this.functionName = line[1]; // we stores the function name for later.
            lineOfCode += 'fn ' + line[1] + line[2];

            for (let word = 3; word < line.length - 2; word++) {
                if (line[word + 1] == ')') {
                    lineOfCode += line[word] + ': ' + variablesType[line[word]];
                } else {
                    lineOfCode += line[word] + ': ' + variablesType[line[word]] + ', ';
                }
            }
            lineOfCode += ')' + ' -> ' + outputType + ' ' + line[line.length - 1];


            lineOfCode += '\n';
            return lineOfCode;
        }

        //If the array contains only one element it means it's a closing bracket '{'.
        else if (line.length == 1) {
            if (line[0] == '{') lineOfCode += '{' + '\n';
            else if (line[0] == '}') lineOfCode += '}' + '\n';
            else lineOfCode += line[0] + '\n';

            return lineOfCode;
        }

        else {
            let skipNextIteration = false; // this boolean is used to skip the name of a variable to avoid repetitions.
            for (let j = 0; j < line.length; j++) {

                if (skipNextIteration) {
                    skipNextIteration = false; // Reset the flag
                    continue; // Skip the rest of this loop iteration
                }

                switch (line[j])
                // replace 'let','var','const' with their equivlant in WGSL
                {
                    case 'let':
                        line[j] = 'var ' + line[j + 1] + ': ' + variablesType[line[j + 1]];
                        skipNextIteration = true;
                        break;
                    case 'var':
                        line[j] = 'var ' + line[j + 1] + ': ' + variablesType[line[j + 1]];
                        skipNextIteration = true;
                        break;
                    case 'const':
                        line[j] = 'let ' + line[j + 1] + ': ' + variablesType[line[j + 1]];
                        skipNextIteration = true;
                        break;
                    // replace return with its equivlant
                    case 'return':
                        line[j] = 'return';
                        break;
                    // replace JS array declaration to WGSL declaration
                    case '[':
                        if (line[j - 1] == '=') // dectect if the line is a declaration of an array
                        {
                            let elements = ''; // to store all array elements
                            let i = j + 1;
                            let num_vir = 0;
                            while (i < line.length) {
                                if (line[i] == ',') {
                                    i++;  //skip the ',' charecter
                                    num_vir++;
                                }
                                else if (line[i] == ']') {
                                    break; // stop when finding ']' charecter (array ends)
                                }
                                else {

                                    elements += line[i] + ',';// count the number of elements 
                                    i++;
                                }

                            }
                            num_vir++;
                            this.variablesOfFunction[line[j - 2]] = "array<f32," + String(num_vir) + ">";

                            elements = elements.slice(0, -1); // remove the last charecter because it is always a ',' look above
                            lineOfCode += 'array(' + elements + ')';
                            return lineOfCode;
                        }
                    case 'switch':
                        line = line.filter(item => item !== ')');
                        line = line.filter(item => item !== '(');
                    case 'Math':
                        line = line.filter(item => item !== 'Math');
                        line[line.indexOf('Math') + 1] = '';// remove the '.' that comes after math module call
                        break;
                    case "\'use":
                        return '';
                }
                if (line[j] == '.' || line[j + 1] == '.') {
                    lineOfCode += line[j];
                } else {
                    lineOfCode += line[j] + ' ';
                }

            }


            lineOfCode += "\n";
            return lineOfCode;

        }



    }


    /**
     * @method shaderCodeUpdate: launches the gneration of the updated shader code if corrections given.
     */
    shaderCodeUpdate() {
        console.log(this.theTranscriptOfFunctions);
        this.shader_code = this.shaderCodeHead;// re-initializes the new shader code with the head.

        Object.keys(this.theTranscriptOfFunctions).forEach(key => {


            let words = this.theTranscriptOfFunctions[key];
            let variablesType = this.functionVariables[key];

            if (key == this.mainFunctionName) {
                for (let x = 0; x < words.length; x++) {

                    this.shader_code += this.mainKernelUpdate(words[x], x, variablesType);
                }
            } else {
                for (let x = 0; x < words.length; x++) {

                    this.shader_code += this.kernelUpdate(words[x], x, variablesType["return"], variablesType);
                }
            }

        });
        console.log(this.shader_code);

    }

    /**
     * @method graphicInterface: generates the interface for the introduction of correction during the validation phase.
     */
    graphicInterface() {
        const dropdownButton = document.querySelector('.dropbtn');
        const dropdownContent = document.getElementById('dropdown-content');
        const dictionaryDetails = document.getElementById('dictionary-details');
        const saveButton = document.getElementById('save-button');

        dropdownButton.addEventListener('click', function () {
            dropdownContent.classList.toggle('show');
        });

        window.addEventListener('click', function (event) {
            if (!event.target.matches('.dropbtn')) {
                if (dropdownContent.classList.contains('show')) {
                    dropdownContent.classList.remove('show');
                }
            }
        });

        const createDropdownItems = (dictionary) => {
            for (const key in dictionary) {
                if (dictionary.hasOwnProperty(key)) {
                    const item = document.createElement('a');
                    item.href = '#';
                    item.innerText = key;
                    item.addEventListener('click', function (event) {
                        event.preventDefault();
                        selectItem(key, dictionary);
                    });
                    dropdownContent.appendChild(item);
                }
            }
        }

        const selectItem = (key, dictionary) => {
            dictionaryDetails.innerHTML = ''; // Clear previous details
            const details = dictionary[key];

            for (const detailKey in details) {
                if (details.hasOwnProperty(detailKey)) {
                    const entryDiv = document.createElement('div');
                    entryDiv.className = 'entry';

                    const label = document.createElement('label');
                    label.innerText = `${detailKey}:`;

                    const input = document.createElement('input');
                    input.type = 'text';
                    input.value = details[detailKey];
                    input.dataset.key = detailKey;
                    input.dataset.parent = key;

                    entryDiv.appendChild(label);
                    entryDiv.appendChild(input);
                    dictionaryDetails.appendChild(entryDiv);
                }
            }

            saveButton.style.display = 'block';
        }


        const saveChanges = (dictionary) => {
            const dictionaryDetails = document.getElementById('dictionary-details');
            const inputs = dictionaryDetails.querySelectorAll('input');
            inputs.forEach(input => {
                const parentKey = input.dataset.parent;
                const detailKey = input.dataset.key;
                dictionary[parentKey][detailKey] = input.value;
            });

            alert('Changes saved!');
            console.log(dictionary);
        }

        saveButton.addEventListener('click', () => {
            saveChanges(this.functionVariables);
            this.variableTypesChanged = true;
        });

        // Initialize dropdown  
        createDropdownItems(this.functionVariables);
    }

    /**
     * @method waitForConfirmation: stops the execution giving time for correction, resumes execution upon clicking on the button confirm.
     * @returns : continuation of the execution.
     */
    waitForConfirmation() {
        return new Promise((resolve) => {
            const confirmButton = document.getElementById('confirm-button');

            // Event listener to resolve the promise when the button is clicked
            confirmButton.addEventListener('click', () => {
                if (this.variableTypesChanged) {
                    this.shaderCodeUpdate();
                }
                resolve();
            }, { once: true });  // { once: true } ensures the event listener is removed after it is triggered
        });
    }


    /**
     * @method redirectConsoleLog: Function to redirect console.log output to the right frame.
     */
    redirectConsoleLog() {
        const originalLog = console.log;
        const rightFrame = document.getElementById('right-frame');

        console.log = function () {
            // Capture arguments passed to console.log
            const args = Array.from(arguments);

            // Format the arguments for display (optional)
            //const formattedMessage = args.join(';<br>'); // Join arguments with spaces

            // Create a new paragraph element with the message
            const messageElement = document.createElement('pre');
            messageElement.textContent = args;

            // Append the paragraph element to the right frame
            rightFrame.appendChild(messageElement);

            // Call the original console.log for debugging purposes (optional)
            originalLog.apply(console, args);
        };
    }


    /**
     * @method processMain: used to translate the main function.
     * @param {function} func 
     * @param {Array} workGroupsSize 
     */
    processMain(func, workGroupsSize) {

        const funcString = func.toString();// Convert the function to a string.
        const lines = funcString.split("\n");// cut the resulted string into lines.
        const delimiters = /[\s]/;// we prepare the specific delimiters to use to clean the lines (here we have : spaces and ',' and ';').

        for (let x = 0; x < lines.length; x++) {
            let word = lines[x].split(delimiters);// we clean the selected line with the chosen delemitters.

            let newArr = [];

            // this part is just to separate the '(' and ')' from the arguments 
            word.forEach(entry => {
                let parts = entry.split(/(\(|\)|\[|\]|\,|\.)/); // Split where '(' is preceded by a space or at the start of the string
                // Add each part to the new array
                parts.forEach(part => {
                    newArr.push(part.trim()); // Trim to remove any leading or trailing whitespace
                });
            });



            word = newArr.filter(entry => entry);// we filters our array to take out all the empty entries

            this.functionTextAsArray.push(word);
            //shader_code += word.join(' ') + '\n';
            this.shader_code += this.mainFunctionKernel(word, x, workGroupsSize) + '\n';// a call to the kernel procedure that translates the code line.

        }


        this.functionVariables[this.functionName] = this.variablesOfFunction;// we save the information about each function here.
        this.theTranscriptOfFunctions[this.functionName] = this.functionTextAsArray;
        this.inputVariablesDict = {};// we empty this Object for future uses.
        this.variablesOfFunction = {};// we empty this Object for future uses.
        this.functionTextAsArray = [];// we empty the array for future uses.


        // output the resulted translated function.
        console.log(this.shader_code);

    }

    /**
     * @method mainFunctionKernel: called upon by the ProcessMain to tranlate the main function line by line.
     * @param {String} line : a line of code in string format.
     * @param {Number} x : the number of the line.
     * @param {Array} workGroupsSize : the workgroupsize in array format.
     * @returns {String}: a line of code tranlated.
     */
    mainFunctionKernel(line, x, workGroupsSize) {
        let lineOfCode = '';
        for (let x = 0; x < workGroupsSize.length; x++) {
            this.workgroups_size *= workGroupsSize[x];
        }
        //Here we transform the first line of the function (the function declaration).
        if (x == 0) {
            line = line.filter(item => item !== ',')
            const values = Object.values(this.inputVariablesDict);
            this.functionName = line[1]; // we stores the function name for later.
            this.mainFunctionName = line[1];
            lineOfCode += "@compute @workgroup_size(" + String(this.workgroups_size) + ") " + 'fn ' + line[1] + line[2];

            lineOfCode += " @builtin(global_invocation_id) id: vec3<u32>) " + line[line.length - 1];


            lineOfCode += '\n';
            return lineOfCode;
        }

        //If the array contains only one element it means it's a closing bracket '{'.
        else if (line.length == 1) {
            if (line[0] == '{') lineOfCode += '{' + '\n';
            else if (line[0] == '}') lineOfCode += '}' + '\n';
            else lineOfCode += line[0] + '\n';

            return lineOfCode;
        }

        else {
            for (let j = 0; j < line.length; j++) {
                switch (line[j])
                // replace 'let','var','const' with their equivlant in WGSL
                {
                    case 'let':
                        this.variablesOfFunction[line[j + 1]] = "f32";
                        line[j] = 'var';
                        break;
                    case 'var':
                        this.variablesOfFunction[line[j + 1]] = "f32";
                        line[j] = 'var';
                        break;
                    case 'const':
                        this.variablesOfFunction[line[j + 1]] = "f32";
                        line[j] = 'let';
                        break;
                    // replace return with its equivlant
                    case 'return':
                        line[j] = 'return';
                        break;
                    // replace JS array declaration to WGSL declaration
                    case '[':
                        if (line[j - 1] == '=') // dectect if the line is a declaration of an array
                        {
                            let elements = ''; // to store all array elements
                            let i = j + 1;
                            let num_vir = 0;
                            while (i < line.length) {
                                if (line[i] == ',') {
                                    i++;  //skip the ',' charecter
                                    num_vir++;
                                }
                                else if (line[i] == ']') {
                                    break; // stop when finding ']' charecter (array ends)
                                }
                                else {

                                    elements += line[i] + ',';// count the number of elements 
                                    i++;
                                }

                            }
                            num_vir++;
                            this.variablesOfFunction[line[j - 2]] = "array<f32," + String(num_vir) + ">";

                            elements = elements.slice(0, -1); // remove the last charecter because it is always a ',' look above
                            lineOfCode += 'array(' + elements + ')';
                            return lineOfCode;
                        }
                    case 'switch':
                        line = line.filter(item => item !== ')');
                        line = line.filter(item => item !== '(');
                    case 'Math':
                        line = line.filter(item => item !== 'Math');
                        line[line.indexOf('Math') + 1] = '';// remove the '.' that comes after math module call
                        break;
                    case "\'use":
                        return '';
                }
                if (line[j] == '.' || line[j + 1] == '.') {
                    lineOfCode += line[j];
                } else {
                    lineOfCode += line[j] + ' ';
                }

            }


            lineOfCode += "\n";
            return lineOfCode;

        }
    }

    /**
     * @method mainKernelUpdate: called upon to re-translate the main fuction if changes where introduced.
     * @param {String} line : a line of code in string format.
     * @param {Number} x : the number of the line.
     * @param {Object} variablesType: an Object containing the types of all variables. 
     * @returns 
     */
    mainKernelUpdate(line, x, variablesType) {
        let lineOfCode = '';

        //Here we transform the first line of the function (the function declaration).
        if (x == 0) {
            line = line.filter(item => item !== ',')

            lineOfCode += "@compute @workgroup_size(" + String(this.workgroups_size) + ") " + 'fn ' + line[1] + line[2];

            lineOfCode += " @builtin(global_invocation_id) id: vec3<u32>) " + line[line.length - 1];


            lineOfCode += '\n';
            return lineOfCode;
        }

        //If the array contains only one element it means it's a closing bracket '{'.
        else if (line.length == 1) {
            if (line[0] == '{') lineOfCode += '{' + '\n';
            else if (line[0] == '}') lineOfCode += '}' + '\n';
            else lineOfCode += line[0] + '\n';

            return lineOfCode;
        }

        else {
            let skipNextIteration = false; // this boolean is used to skip the name of a variable to avoid repetitions.
            for (let j = 0; j < line.length; j++) {

                if (skipNextIteration) {
                    skipNextIteration = false; // Reset the flag
                    continue; // Skip the rest of this loop iteration
                }

                switch (line[j])
                // replace 'let','var','const' with their equivlant in WGSL
                {
                    case 'let':
                        line[j] = 'var ' + line[j + 1] + ': ' + variablesType[line[j + 1]];
                        skipNextIteration = true;
                        break;
                    case 'var':
                        line[j] = 'var ' + line[j + 1] + ': ' + variablesType[line[j + 1]];
                        skipNextIteration = true;
                        break;
                    case 'const':
                        line[j] = 'let ' + line[j + 1] + ': ' + variablesType[line[j + 1]];
                        skipNextIteration = true;
                        break;
                    // replace return with its equivlant
                    case 'return':
                        line[j] = 'return';
                        break;
                    // replace JS array declaration to WGSL declaration
                    case '[':
                        if (line[j - 1] == '=') // dectect if the line is a declaration of an array
                        {
                            let elements = ''; // to store all array elements
                            let i = j + 1;
                            let num_vir = 0;
                            while (i < line.length) {
                                if (line[i] == ',') {
                                    i++;  //skip the ',' charecter
                                    num_vir++;
                                }
                                else if (line[i] == ']') {
                                    break; // stop when finding ']' charecter (array ends)
                                }
                                else {

                                    elements += line[i] + ',';// count the number of elements 
                                    i++;
                                }

                            }
                            num_vir++;
                            this.variablesOfFunction[line[j - 2]] = "array<f32," + String(num_vir) + ">";

                            elements = elements.slice(0, -1); // remove the last charecter because it is always a ',' look above
                            lineOfCode += 'array(' + elements + ')';
                            return lineOfCode;
                        }
                    case 'switch':
                        line = line.filter(item => item !== ')');
                        line = line.filter(item => item !== '(');
                    case 'Math':
                        line = line.filter(item => item !== 'Math');
                        line[line.indexOf('Math') + 1] = '';// remove the '.' that comes after math module call
                        break;
                    case "\'use":
                        return '';
                }
                if (line[j] == '.' || line[j + 1] == '.') {
                    lineOfCode += line[j];
                } else {
                    lineOfCode += line[j] + ' ';
                }

            }


            lineOfCode += "\n";
            return lineOfCode;

        }
    }
}


function add(a, b) {
    let c = 5;
    while (c < 20) {
        c += 12;
    }
    return c;
}

function sub(x, y) {
    return x;
}

function main_kernel() {
    let i = id.x;
    myBuffer[i] = myBuffer[i] * 2;
    let A = 0;
    var B = 15;

}

// Usage example
async function main() {

    // Assuming you have already obtained an adapter and device


    const webGPUManager = new WebGPUManager();
    await webGPUManager.initialization();

    // Your data to copy into the buffer
    const data = new Float32Array([0.0, -1.0, -2.0, -3.0]);
    const data2 = new Float32Array([0.0, 1.0, 2.0, 3.0]);

    inputData = [
        ['myBuffer', data],
        ['myBuffer2', data2, 0]
    ];

    for (let x = 0; x < inputData.length; x++) {
        webGPUManager.createInputBuffer(inputData[x][0], inputData[x][1], !inputData[x][2] ? inputData[x][2] : 0);
    }

    outputData = [
        ['result', 'myBuffer', data],
        ['result2', 'myBuffer2', data2]
    ];

    for (let x = 0; x < inputData.length; x++) {
        webGPUManager.createOutputBuffer(outputData[x][0], outputData[x][1], outputData[x][2]);
    }


    document.addEventListener('DOMContentLoaded', function () {



        webGPUManager.createEncoder();
    });


    webGPUManager.processFunction(add, "f32", 3, 4);
    webGPUManager.processFunction(sub, "f32", 3, [4, 4, 5]);
    webGPUManager.processMain(main_kernel, 2);

    webGPUManager.graphicInterface();
    await webGPUManager.waitForConfirmation();
    webGPUManager.createBindGroup('hello');
    webGPUManager.createPipeline('hi');




    webGPUManager.createEncoder('work', data.length);
    webGPUManager.getResult();


}

main();




