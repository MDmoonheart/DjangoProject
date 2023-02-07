// namespaces
var dwv = dwv || {};
dwv.image = dwv.image || {};

/**
 * The JPEG baseline decoder.
 *
 * @external JpegImage
 * @see https://github.com/mozilla/pdf.js/blob/master/src/core/jpg.js
 */
/* global JpegImage */
var hasJpegBaselineDecoder = (typeof JpegImage !== 'undefined');

/**
 * The JPEG decoder namespace.
 *
 * @external jpeg
 * @see https://github.com/rii-mango/JPEGLosslessDecoderJS
 */
/* global jpeg */
var hasJpegLosslessDecoder = (typeof jpeg !== 'undefined') &&
    (typeof jpeg.lossless !== 'undefined');

/**
 * The JPEG 2000 decoder.
 *
 * @external JpxImage
 * @see https://github.com/jpambrun/jpx-medical/blob/master/jpx.js
 */
/* global JpxImage */
var hasJpeg2000Decoder = (typeof JpxImage !== 'undefined');

/**
 * Asynchronous pixel buffer decoder.
 *
 * @class
 * @param {string} script The path to the decoder script to be used
 *   by the web worker.
 * @param {number} _numberOfData The anticipated number of data to decode.
 */
dwv.image.AsynchPixelBufferDecoder = function (script, _numberOfData) {
  // initialise the thread pool
  var pool = new dwv.utils.ThreadPool(10);
  // flag to know if callbacks are set
  var areCallbacksSet = false;
  // closure to self
  var self = this;

  /**
   * Decode a pixel buffer.
   *
   * @param {Array} pixelBuffer The pixel buffer.
   * @param {object} pixelMeta The input meta data.
   * @param {object} info Information object about the input data.
   */
  this.decode = function (pixelBuffer, pixelMeta, info) {
    if (!areCallbacksSet) {
      areCallbacksSet = true;
      // set event handlers
      pool.onworkstart = self.ondecodestart;
      pool.onworkitem = self.ondecodeditem;
      pool.onwork = self.ondecoded;
      pool.onworkend = self.ondecodeend;
      pool.onerror = self.onerror;
      pool.onabort = self.onabort;
    }
    // create worker task
    var workerTask = new dwv.utils.WorkerTask(
      script,
      {
        buffer: pixelBuffer,
        meta: pixelMeta
      },
      info
    );
    // add it the queue and run it
    pool.addWorkerTask(workerTask);
  };

  /**
   * Abort decoding.
   */
  this.abort = function () {
    // abort the thread pool, will trigger pool.onabort
    pool.abort();
  };
};

/**
 * Handle a decode start event.
 * Default does nothing.
 *
 * @param {object} _event The decode start event.
 */
dwv.image.AsynchPixelBufferDecoder.prototype.ondecodestart = function (
  _event) {};
/**
 * Handle a decode item event.
 * Default does nothing.
 *
 * @param {object} _event The decode item event fired
 *   when a decode item ended successfully.
 */
dwv.image.AsynchPixelBufferDecoder.prototype.ondecodeditem = function (
  _event) {};
/**
 * Handle a decode event.
 * Default does nothing.
 *
 * @param {object} _event The decode event fired
 *   when a file has been decoded successfully.
 */
dwv.image.AsynchPixelBufferDecoder.prototype.ondecoded = function (
  _event) {};
/**
 * Handle a decode end event.
 * Default does nothing.
 *
 * @param {object} _event The decode end event fired
 *  when a file decoding has completed, successfully or not.
 */
dwv.image.AsynchPixelBufferDecoder.prototype.ondecodeend = function (
  _event) {};
/**
 * Handle an error event.
 * Default does nothing.
 *
 * @param {object} _event The error event.
 */
dwv.image.AsynchPixelBufferDecoder.prototype.onerror = function (_event) {};
/**
 * Handle an abort event.
 * Default does nothing.
 *
 * @param {object} _event The abort event.
 */
dwv.image.AsynchPixelBufferDecoder.prototype.onabort = function (_event) {};

/**
 * Synchronous pixel buffer decoder.
 *
 * @class
 * @param {string} algoName The decompression algorithm name.
 * @param {number} numberOfData The anticipated number of data to decode.
 */
dwv.image.SynchPixelBufferDecoder = function (algoName, numberOfData) {
  // decode count
  var decodeCount = 0;

  /**
   * Decode a pixel buffer.
   *
   * @param {Array} pixelBuffer The pixel buffer.
   * @param {object} pixelMeta The input meta data.
   * @param {object} info Information object about the input data.
   * @external jpeg
   * @external JpegImage
   * @external JpxImage
   */
  this.decode = function (pixelBuffer, pixelMeta, info) {
    ++decodeCount;

    var decoder = null;
    var decodedBuffer = null;
    if (algoName === 'jpeg-lossless') {
      if (!hasJpegLosslessDecoder) {
        throw new Error('No JPEG Lossless decoder provided');
      }
      // bytes per element
      var bpe = pixelMeta.bitsAllocated / 8;
      var buf = new Uint8Array(pixelBuffer);
      decoder = new jpeg.lossless.Decoder();
      var decoded = decoder.decode(buf.buffer, 0, buf.buffer.byteLength, bpe);
      if (pixelMeta.bitsAllocated === 8) {
        if (pixelMeta.isSigned) {
          decodedBuffer = new Int8Array(decoded.buffer);
        } else {
          decodedBuffer = new Uint8Array(decoded.buffer);
        }
      } else if (pixelMeta.bitsAllocated === 16) {
        if (pixelMeta.isSigned) {
          decodedBuffer = new Int16Array(decoded.buffer);
        } else {
          decodedBuffer = new Uint16Array(decoded.buffer);
        }
      }
    } else if (algoName === 'jpeg-baseline') {
      if (!hasJpegBaselineDecoder) {
        throw new Error('No JPEG Baseline decoder provided');
      }
      decoder = new JpegImage();
      decoder.parse(pixelBuffer);
      decodedBuffer = decoder.getData(decoder.width, decoder.height);
    } else if (algoName === 'jpeg2000') {
      if (!hasJpeg2000Decoder) {
        throw new Error('No JPEG 2000 decoder provided');
      }
      // decompress pixel buffer into Int16 image
      decoder = new JpxImage();
      decoder.parse(pixelBuffer);
      // set the pixel buffer
      decodedBuffer = decoder.tiles[0].items;
    } else if (algoName === 'rle') {
      // decode DICOM buffer
      decoder = new dwv.decoder.RleDecoder();
      // set the pixel buffer
      decodedBuffer = decoder.decode(
        pixelBuffer,
        pixelMeta.bitsAllocated,
        pixelMeta.isSigned,
        pixelMeta.sliceSize,
        pixelMeta.samplesPerPixel,
        pixelMeta.planarConfiguration);
    }
    // send decode events
    this.ondecodeditem({
      data: [decodedBuffer],
      index: info.itemNumber
    });
    // decode end?
    if (decodeCount === numberOfData) {
      this.ondecoded({});
      this.ondecodeend({});
    }
  };

  /**
   * Abort decoding.
   */
  this.abort = function () {
    // nothing to do in the synchronous case.
    // callback
    this.onabort({});
    this.ondecodeend({});
  };
};

/**
 * Handle a decode start event.
 * Default does nothing.
 *
 * @param {object} _event The decode start event.
 */
dwv.image.SynchPixelBufferDecoder.prototype.ondecodestart = function (
  _event) {};
/**
 * Handle a decode item event.
 * Default does nothing.
 *
 * @param {object} _event The decode item event fired
 *   when a decode item ended successfully.
 */
dwv.image.SynchPixelBufferDecoder.prototype.ondecodeditem = function (
  _event) {};
/**
 * Handle a decode event.
 * Default does nothing.
 *
 * @param {object} _event The decode event fired
 *   when a file has been decoded successfully.
 */
dwv.image.SynchPixelBufferDecoder.prototype.ondecoded = function (
  _event) {};
/**
 * Handle a decode end event.
 * Default does nothing.
 *
 * @param {object} _event The decode end event fired
 *  when a file decoding has completed, successfully or not.
 */
dwv.image.SynchPixelBufferDecoder.prototype.ondecodeend = function (
  _event) {};
/**
 * Handle an error event.
 * Default does nothing.
 *
 * @param {object} _event The error event.
 */
dwv.image.SynchPixelBufferDecoder.prototype.onerror = function (_event) {};
/**
 * Handle an abort event.
 * Default does nothing.
 *
 * @param {object} _event The abort event.
 */
dwv.image.SynchPixelBufferDecoder.prototype.onabort = function (_event) {};

/**
 * Decode a pixel buffer.
 *
 * @class
 * @param {string} algoName The decompression algorithm name.
 * @param {number} numberOfData The anticipated number of data to decode.
 * If the 'dwv.image.decoderScripts' variable does not contain the desired,
 * algorythm the decoder will switch to the synchronous mode.
 */
dwv.image.PixelBufferDecoder = function (algoName, numberOfData) {
  /**
   * Pixel decoder.
   * Defined only once.
   *
   * @private
   * @type {object}
   */
  var pixelDecoder = null;

  // initialise the asynch decoder (if possible)
  if (typeof dwv.image.decoderScripts !== 'undefined' &&
    typeof dwv.image.decoderScripts[algoName] !== 'undefined') {
    pixelDecoder = new dwv.image.AsynchPixelBufferDecoder(
      dwv.image.decoderScripts[algoName], numberOfData);
  } else {
    pixelDecoder = new dwv.image.SynchPixelBufferDecoder(
      algoName, numberOfData);
  }

  // flag to know if callbacks are set
  var areCallbacksSet = false;

  /**
   * Get data from an input buffer using a DICOM parser.
   *
   * @param {Array} pixelBuffer The input data buffer.
   * @param {object} pixelMeta The input meta data.
   * @param {object} info Information object about the input data.
   */
  this.decode = function (pixelBuffer, pixelMeta, info) {
    if (!areCallbacksSet) {
      areCallbacksSet = true;
      // set callbacks
      pixelDecoder.ondecodestart = this.ondecodestart;
      pixelDecoder.ondecodeditem = this.ondecodeditem;
      pixelDecoder.ondecoded = this.ondecoded;
      pixelDecoder.ondecodeend = this.ondecodeend;
      pixelDecoder.onerror = this.onerror;
      pixelDecoder.onabort = this.onabort;
    }
    // decode and call the callback
    pixelDecoder.decode(pixelBuffer, pixelMeta, info);
  };

  /**
   * Abort decoding.
   */
  this.abort = function () {
    // decoder classes should define an abort
    pixelDecoder.abort();
  };
};

/**
 * Handle a decode start event.
 * Default does nothing.
 *
 * @param {object} _event The decode start event.
 */
dwv.image.PixelBufferDecoder.prototype.ondecodestart = function (_event) {};
/**
 * Handle a decode item event.
 * Default does nothing.
 *
 * @param {object} _event The decode item event fired
 *   when a decode item ended successfully.
 */
dwv.image.PixelBufferDecoder.prototype.ondecodeditem = function (_event) {};
/**
 * Handle a decode event.
 * Default does nothing.
 *
 * @param {object} _event The decode event fired
 *   when a file has been decoded successfully.
 */
dwv.image.PixelBufferDecoder.prototype.ondecoded = function (_event) {};
/**
 * Handle a decode end event.
 * Default does nothing.
 *
 * @param {object} _event The decode end event fired
 *  when a file decoding has completed, successfully or not.
 */
dwv.image.PixelBufferDecoder.prototype.ondecodeend = function (_event) {};
/**
 * Handle an error event.
 * Default does nothing.
 *
 * @param {object} _event The error event.
 */
dwv.image.PixelBufferDecoder.prototype.onerror = function (_event) {};
/**
 * Handle an abort event.
 * Default does nothing.
 *
 * @param {object} _event The abort event.
 */
dwv.image.PixelBufferDecoder.prototype.onabort = function (_event) {};
