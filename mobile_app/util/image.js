const fs = require('node:fs');
const { type } = require('node:os');

const getImages = function (folder) {
    let files = fs.readdirSync(folder, { withFileTypes: true, recursive: true })

    let imageList = []
    for (let i = 0; i < files.length; i++) {
        if (files[i].isFile() && isImage(files[i].name)) {
            imageList.push(files[i].parentPath + '/' + files[i].name)
        }
    }

    return imageList
};

function isImage(fileName) {
    const types = ['.png', '.jpg', '.jpeg']

    for (let i = 0; i < types.length; i++) {
        if (fileName.indexOf(types[i]) > 0) {
            return true
        }
    }

    return false
};

module.exports = {
    getImages: getImages
};