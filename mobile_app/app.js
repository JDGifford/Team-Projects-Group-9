const express = require('express')
const images = require('./util/image.js')

const app = express()
const port = 3000

// Set Pug as the template rendering engine
app.set('view engine', 'pug')

// const imageData = ['./images/IMG_20240729_124341114.jpg', '../images/Death_Star_Plans4.png']

app.get('/', (req, res) => {
    // Get a list of the image files
    const imageList = images.getImages('./public/images')

    for (let i = 0; i < imageList.length; i++) {
        imageList[i] = imageList[i].replace("./public", "./")
    }

    res.render('index', { imageList })
})

app.use(express.static('public'))
// app.use(express.static('images'))

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})

