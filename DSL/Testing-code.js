class WebGPUManager {
    /**
     * contructor represents a method that runs when a new intance of the class is created.
     */
    constructor(device) {
        this.device = device;// the GPU device that will runs our WGSL code.
        this.inputBuffers = {};// a dictionnary that stores the names and refferences of the input data (variables).
        this.outputBuffers = {};// a dictionnary that stores the names and refferences of the output data (variables).
        this.linkedVariables = {};// a dictionnary that stores the link between the input and output variables (from which input buffer will the result buffer copy data).
        this.functionArguments = {};
        this.shader_code = '';// a string that will hold the translated function (WGSL code).
        this.counter = 0; // a counter for the bindings of the same group.
    }

    /**
     * @method createInputBuffer: represent a method used to copy data into the GPU for later use
     * @param {string} name : a string that represents the name the variable will have in the WGSL code.
     * @param {same as the input variable} data : the input data to be copied into the GPU.
     */
    createInputBuffer(inputBuffersArray) {
        for (let x = 0; x < inputBuffersArray.length; x++) {
            const data = inputBuffersArray[x][1];
            const name = inputBuffersArray[x][0];
            const group_binding = inputBuffersArray[x][2];
            const dataSize = data.byteLength;
            const buffer = this.device.createBuffer({
                size: dataSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                mappedAtCreation: false
            });
            // We write the data into the created buffer.
            this.device.queue.writeBuffer(buffer, 0, data);

            this.inputBuffers[name] = buffer; // fill ou the dictionnary { name: buffer refference }.

            this.shader_code += '@group(' + String(group_binding) + ') ' + '@binding('
                + String(this.counter) + ') var<storage, read_write> ' + name + ' : array<f32>;\n'
            this.counter++;
        }
    }

    /**
     * @function createOutputBuffer: represents a method to get results from the GPU after computation.
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

        this.outputBuffers[name] = buffer;// fill ou the dictionnary { name: buffer refference }.
        this.linkedVariables[name] = linked_variable;// fill ou the dictionnary { name: linked_variable }.
    }



    createPipeline(description) {
        this.shader_code += `@compute @workgroup_size(1) fn computeSomething(
        @builtin(global_invocation_id) id: vec3<u32>
      ) {
        let i = id.x;
        myBuffer[i] = myBuffer[i] * 2;
        
        myBuffer2[i] = add(myBuffer[i], d);
      }`
        const module = this.device.createShaderModule({
            label: description + ' module',
            code: this.shader_code,
        });

        this.pipeline = this.device.createComputePipeline({
            label: description,
            layout: 'auto',
            compute: {
                module,
                entryPoint: 'computeSomething',
            },
        });
    }

    /**
     * @method createBindGroup : a method that generates a bindGroup automatically when created 
     * @param {string} description : a short string referencing the bindgroup to be created, very usefull as a flag in case of errors.
     */
    createBindGroup(description) {

        const bindings = [];// An array that will stores the format of our bindings 

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
                i++;
            }
        }


        // Creations of the binding group
        this.bindGroup = this.device.createBindGroup({
            label: description,
            layout: this.pipeline.getBindGroupLayout(0),// this here needs to be changer when we define our pipeline Layout.
            entries: bindings,
        });

    }

    /**
     * @method createEncoder : Sets up the commands that are to be executed by the GPU upon call.
     * @param {string} description : a short label depecting our encoder.
     * @param {vec3} workGroupsSize : a 3D arrays that defines the number of independents threads to be executed.
     */
    createEncoder(description, workGroupsSize) {
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
            lineOfCode += 'fn ' + line[1] + line[2];

            let index = 0;
            for (let word = 3; word < line.length - 2; word++) {
                if (line[word + 1] == ')') {
                    lineOfCode += line[word] + ': ' + values[index];
                } else {
                    lineOfCode += line[word] + ': ' + values[index] + ', ';
                    index += 1;
                }
            }
            lineOfCode += ')' + ' -> ' + outputType + ' ' + line[line.length - 1];

            lineOfCode += '\n';
            return lineOfCode
        }

        //If the array contains only one element it means it's a closing bracket '{'.
        else if (line.length == 1) {
            if (line[0] == '{') lineOfCode += '{' + '\n';
            else if (line[0] == '}') lineOfCode += '}' + '\n';
            else lineOfCode += line[0] + '\n'

            return lineOfCode
        }

        else {
            for (let j = 0; j < line.length; j++) {
                switch (line[j])
                // replace 'let','var','const' with their equivlant in WGSL
                {
                    case 'let':
                        line[j] = 'var '
                        break
                    case 'var':
                        line[j] = 'var '
                        break
                    case 'const':
                        line[j] = 'let '
                        break
                    // replace return with its equivlant
                    case 'return': {
                        line[j] = 'return '
                    }
                    // replace JS array declaration to WGSL declaration
                    case '[':
                        if (line[j - 1] == '=') // dectect if the line is a declaration of an array
                        {
                            let elements = '' // to store all array elements
                            let i = j + 1
                            while (i < line.length) {
                                if (line[i] == ',') {
                                    i++  //skip the ',' charecter
                                }
                                else if (line[i] == ']') {
                                    break // stop when finding ']' charecter (array ends)
                                }
                                else {

                                    elements += line[i] + ',' // count the number of elements 
                                    i++
                                }

                            }
                            elements = elements.slice(0, -1) // remove the last charecter because it is always a ',' look above
                            lineOfCode += 'array(' + elements + ')'
                            return lineOfCode
                        }
                    case 'switch':
                        line = line.filter(item => item !== ')');
                        line = line.filter(item => item !== '(');
                }
                lineOfCode += line[j] + ' '
            }
            lineOfCode += "\n"
            return lineOfCode

        }



    }




    /**
     * @method processFunction : a method that pre-process the input function then pass it into the kernel for convertion.
     * @param {function} func : the call of the javascript function to be translated into WGSL. 
     * @param {string} outputType : a string representing the type of the output of the function.
     * @param  {...any} args : the parameters of the function, inputed as in a normall call to this function.
     */
    processFunction(func, outputType, ...args) {
        this.inputVariablesDict = {};//this dictionnary stores the parameters of the function and their type (in WGSL).

        // Here we fill the inputVariablesDict with the javascript type of each parameter.
        for (let x = 0; x < args.length; x++) {
            this.inputVariablesDict[args[x]] = typeof (args[x]);
        }
        console.log(this.inputVariablesDict);// just to verify the correct output

        // Now we need to translate the type of each arguments into the corresponding type in WGSL (this needs more thinking)
        Object.keys(this.inputVariablesDict).forEach(key => {
            if (this.inputVariablesDict[key] == 'number') {
                this.inputVariablesDict[key] = 'f32';
            } else if (this.inputVariablesDict[key] == 'object') {
                this.inputVariablesDict[key] = 'array<f32,2>';
            }
            console.log(key);
        });

        // Simple printing.
        Object.keys(this.inputVariablesDict).forEach(key => {
            console.log(this.inputVariablesDict[key]);
        });


        const funcString = func.toString();// Convert the function to a string.
        const lines = funcString.split("\n");// cut the resulted string into lines.
        const delimiters = /[\s]/;// we prepare the specific delimiters to use to clean the lines (here we have : spaces and ',' and ';').


        for (let x = 0; x < lines.length; x++) {
            let word = lines[x].split(delimiters);// we clean the selected line with the chosen delemitters.

            let newArr = [];

            // this part is just to separate the '(' and ')' from the arguments 
            word.forEach(entry => {
                let parts = entry.split(/(\(|\)|\[|\]|\,)/); // Split where '(' is preceded by a space or at the start of the string
                // Add each part to the new array
                parts.forEach(part => {
                    newArr.push(part.trim()); // Trim to remove any leading or trailing whitespace
                });
            });



            word = newArr.filter(entry => entry);// we filters our array to take out all the empty entries


            if (x == 0) {
                const functionName = word[2];
            }

            //shader_code += word.join(' ') + '\n';
            this.shader_code += this.kernel(word, x, outputType) + '\n';// a call to the kernel procedure that translates the code line.
        }

        this.functionArguments = { functionName: this.inputVariablesDict };// we save the information about each function here.
        this.inputVariablesDict = {};// we empty this dictionnary for future uses.

        // output the resulted translated function.
        console.log("Function as string:", this.shader_code);

    }



}
function add(a, b) {

    let c = b[0];
    for (let x = 0; x < 10; x++) {
        c += 5.0;
    }
    return c;
}
// Usage example
async function main() {
    // Assuming you have already obtained an adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const webGPUManager = new WebGPUManager(device);

    // Your data to copy into the buffer
    const data = new Float32Array([0.0, -1.0, -2.0, -3.0]);
    const data2 = new Float32Array([0.0, 1.0, 2.0, 3.0]);

    inputData = [
        ['myBuffer', data, 0],
        ['muBuffer2', data2, 0]
    ];
    // Create the buffer
    webGPUManager.createInputBuffer(inputData);// ARRAY.



    webGPUManager.createOutputBuffer('result', 'myBuffer', data);
    webGPUManager.createOutputBuffer('result2', 'myBuffer2', data2);
    b = [4, 5];
    webGPUManager.processFunction(add, "f32", 3, b);

    webGPUManager.createPipeline('hi');
    webGPUManager.createBindGroup('hello');
    webGPUManager.createEncoder('work', data.length);
    webGPUManager.getResult();

    // Later, you can retrieve the buffer when needed
    console.log(webGPUManager.functionArguments);
}

main();
