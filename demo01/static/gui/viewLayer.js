// namespaces
var dwv = dwv || {};
dwv.gui = dwv.gui || {};

/**
 * View layer.
 *
 * @param {object} containerDiv The layer div, its id will be used
 *   as this layer id.
 * @class
 */
dwv.gui.ViewLayer = function (containerDiv) {

  // specific css class name
  containerDiv.className += ' viewLayer';

  // closure to self
  var self = this;

  /**
   * The view controller.
   *
   * @private
   * @type {object}
   */
  var viewController = null;

  /**
   * The main display canvas.
   *
   * @private
   * @type {object}
   */
  var canvas = null;
  /**
   * The offscreen canvas: used to store the raw, unscaled pixel data.
   *
   * @private
   * @type {object}
   */
  var offscreenCanvas = null;
  /**
   * The associated CanvasRenderingContext2D.
   *
   * @private
   * @type {object}
   */
  var context = null;

  /**
   * Flag to know if the context has been cleared.
   *
   * @private
   * @type {boolean}
   */
  var isContextClear = true;

  /**
   * The image data array.
   *
   * @private
   * @type {Array}
   */
  var imageData = null;

  /**
   * The layer base size as {x,y}.
   *
   * @private
   * @type {object}
   */
  var baseSize;

  /**
   * The layer base spacing as {x,y}.
   *
   * @private
   * @type {object}
   */
  var baseSpacing;

  /**
   * The layer opacity.
   *
   * @private
   * @type {number}
   */
  var opacity = 1;

  /**
   * The layer scale.
   *
   * @private
   * @type {object}
   */
  var scale = {x: 1, y: 1};

  /**
   * The layer fit scale.
   *
   * @private
   * @type {object}
   */
  var fitScale = {x: 1, y: 1};

  /**
   * The layer offset.
   *
   * @private
   * @type {object}
   */
  var offset = {x: 0, y: 0};

  /**
   * The base layer offset.
   *
   * @private
   * @type {object}
   */
  var baseOffset = {x: 0, y: 0};

  /**
   * Data update flag.
   *
   * @private
   * @type {boolean}
   */
  var needsDataUpdate = null;

  /**
   * The associated data index.
   *
   * @private
   * @type {number}
   */
  var dataIndex = null;

  /**
   * Get the associated data index.
   *
   * @returns {number} The index.
   */
  this.getDataIndex = function () {
    return dataIndex;
  };

  /**
   * Listener handler.
   *
   * @private
   * @type {object}
   */
  var listenerHandler = new dwv.utils.ListenerHandler();

  /**
   * Set the associated view.
   *
   * @param {object} view The view.
   */
  this.setView = function (view) {
    // local listeners
    view.addEventListener('wlchange', onWLChange);
    view.addEventListener('colourchange', onColourChange);
    view.addEventListener('positionchange', onPositionChange);
    view.addEventListener('alphafuncchange', onAlphaFuncChange);
    // view events
    for (var j = 0; j < dwv.image.viewEventNames.length; ++j) {
      view.addEventListener(dwv.image.viewEventNames[j], fireEvent);
    }
    // create view controller
    viewController = new dwv.ctrl.ViewController(view);
  };

  /**
   * Get the view controller.
   *
   * @returns {object} The controller.
   */
  this.getViewController = function () {
    return viewController;
  };

  /**
   * Get the canvas image data.
   *
   * @returns {object} The image data.
   */
  this.getImageData = function () {
    return imageData;
  };

  /**
   * Handle an image change event.
   *
   * @param {object} event The event.
   */
  this.onimagechange = function (event) {
    // event.value = [index, image]
    if (dataIndex === event.value[0]) {
      viewController.setImage(event.value[1]);
      needsDataUpdate = true;
    }
  };

  // common layer methods [start] ---------------

  /**
   * Get the id of the layer.
   *
   * @returns {string} The string id.
   */
  this.getId = function () {
    return containerDiv.id;
  };

  /**
   * Get the layer base size (without scale).
   *
   * @returns {object} The size as {x,y}.
   */
  this.getBaseSize = function () {
    return baseSize;
  };

  /**
   * Get the layer opacity.
   *
   * @returns {number} The opacity ([0:1] range).
   */
  this.getOpacity = function () {
    return opacity;
  };

  /**
   * Set the layer opacity.
   *
   * @param {number} alpha The opacity ([0:1] range).
   */
  this.setOpacity = function (alpha) {
    opacity = Math.min(Math.max(alpha, 0), 1);

    /**
     * Opacity change event.
     *
     * @event dwv.App#opacitychange
     * @type {object}
     * @property {string} type The event type.
     */
    var event = {
      type: 'opacitychange',
      value: [opacity]
    };
    fireEvent(event);
  };

  /**
   * Set the layer scale.
   *
   * @param {object} newScale The scale as {x,y}.
   */
  this.setScale = function (newScale) {
    var helper = viewController.getPlaneHelper();
    var orientedNewScale = helper.getOrientedXYZ(newScale);
    scale = {
      x: fitScale.x * orientedNewScale.x,
      y: fitScale.y * orientedNewScale.y
    };
  };

  /**
   * Set the base layer offset. Resets the layer offset.
   *
   * @param {object} off The offset as {x,y}.
   */
  this.setBaseOffset = function (off) {
    var helper = viewController.getPlaneHelper();
    baseOffset = helper.getPlaneOffsetFromOffset3D({
      x: off.getX(),
      y: off.getY(),
      z: off.getZ()
    });
    // reset offset
    offset = baseOffset;
  };

  /**
   * Set the layer offset.
   *
   * @param {object} newOffset The offset as {x,y}.
   */
  this.setOffset = function (newOffset) {
    var helper = viewController.getPlaneHelper();
    var planeNewOffset = helper.getPlaneOffsetFromOffset3D(newOffset);
    offset = {
      x: baseOffset.x + planeNewOffset.x,
      y: baseOffset.y + planeNewOffset.y
    };
  };

  /**
   * Transform a display position to an index.
   *
   * @param {number} x The X position.
   * @param {number} y The Y position.
   * @returns {dwv.math.Index} The equivalent index.
   */
  this.displayToPlaneIndex = function (x, y) {
    var planePos = this.displayToPlanePos(x, y);
    return new dwv.math.Index([
      Math.floor(planePos.x),
      Math.floor(planePos.y)
    ]);
  };

  /**
   * Remove scale from a display position.
   *
   * @param {number} x The X position.
   * @param {number} y The Y position.
   * @returns {object} The de-scaled position as {x,y}.
   */
  this.displayToPlaneScale = function (x, y) {
    return {
      x: x / scale.x,
      y: y / scale.y
    };
  };

  /**
   * Get a plane position from a display position.
   *
   * @param {number} x The X position.
   * @param {number} y The Y position.
   * @returns {object} The plane position as {x,y}.
   */
  this.displayToPlanePos = function (x, y) {
    var deScaled = this.displayToPlaneScale(x, y);
    return {
      x: deScaled.x + offset.x,
      y: deScaled.y + offset.y
    };
  };

  /**
   * Get a main plane position from a display position.
   *
   * @param {number} x The X position.
   * @param {number} y The Y position.
   * @returns {object} The main plane position as {x,y}.
   */
  this.displayToMainPlanePos = function (x, y) {
    var planePos = this.displayToPlanePos(x, y);
    return {
      x: planePos.x - baseOffset.x,
      y: planePos.y - baseOffset.y
    };
  };

  /**
   * Display the layer.
   *
   * @param {boolean} flag Whether to display the layer or not.
   */
  this.display = function (flag) {
    containerDiv.style.display = flag ? '' : 'none';
  };

  /**
   * Check if the layer is visible.
   *
   * @returns {boolean} True if the layer is visible.
   */
  this.isVisible = function () {
    return containerDiv.style.display === '';
  };

  /**
   * Draw the content (imageData) of the layer.
   * The imageData variable needs to be set
   *
   * @fires dwv.App#renderstart
   * @fires dwv.App#renderend
   */
  this.draw = function () {
    /**
     * Render start event.
     *
     * @event dwv.App#renderstart
     * @type {object}
     * @property {string} type The event type.
     */
    var event = {
      type: 'renderstart',
      layerid: this.getId()
    };
    fireEvent(event);

    // update data if needed
    if (needsDataUpdate) {
      updateImageData();
    }

    // context opacity
    context.globalAlpha = opacity;

    // clear context
    this.clear();

    // draw the cached canvas on the context
    // transform takes as input a, b, c, d, e, f to create
    // the transform matrix (column-major order):
    // [ a c e ]
    // [ b d f ]
    // [ 0 0 1 ]
    context.setTransform(
      scale.x,
      0,
      0,
      scale.y,
      -1 * offset.x * scale.x,
      -1 * offset.y * scale.y
    );

    // disable smoothing (set just before draw, could be reset by resize)
    context.imageSmoothingEnabled = false;
    // draw image
    context.drawImage(offscreenCanvas, 0, 0);

    // set clear flag
    isContextClear = false;

    /**
     * Render end event.
     *
     * @event dwv.App#renderend
     * @type {object}
     * @property {string} type The event type.
     */
    event = {
      type: 'renderend',
      layerid: this.getId()
    };
    fireEvent(event);
  };

  /**
   * Initialise the layer: set the canvas and context
   *
   * @param {object} size The image size as {x,y}.
   * @param {object} spacing The image spacing as {x,y}.
   * @param {number} index The associated data index.
   */
  this.initialise = function (size, spacing, index) {
    // set locals
    baseSize = size;
    baseSpacing = spacing;
    dataIndex = index;

    // create canvas
    // (canvas size is set in fitToContainer)
    canvas = document.createElement('canvas');
    containerDiv.appendChild(canvas);

    // check that the getContext method exists
    if (!canvas.getContext) {
      alert('Error: no canvas.getContext method.');
      return;
    }
    // get the 2D context
    context = canvas.getContext('2d');
    if (!context) {
      alert('Error: failed to get the 2D context.');
      return;
    }

    // check canvas
    if (!dwv.gui.canCreateCanvas(baseSize.x, baseSize.y)) {
      throw new Error('Cannot create canvas ' + baseSize.x + ', ' + baseSize.y);
    }

    // off screen canvas
    offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = baseSize.x;
    offscreenCanvas.height = baseSize.y;
    // original empty image data array
    context.clearRect(0, 0, baseSize.x, baseSize.y);
    imageData = context.createImageData(baseSize.x, baseSize.y);

    // update data on first draw
    needsDataUpdate = true;
  };

  /**
   * Fit the layer to its parent container.
   *
   * @param {number} fitScale1D The 1D fit scale.
   * @param {object} fitSize The fit size as {x,y}.
   */
  this.fitToContainer = function (fitScale1D, fitSize) {
    // update fit scale
    fitScale = {
      x: fitScale1D * baseSpacing.x,
      y: fitScale1D * baseSpacing.y
    };
    // new canvas size
    var width = fitSize.x;
    var height = fitSize.y;
    if (!dwv.gui.canCreateCanvas(width, height)) {
      throw new Error('Cannot resize canvas ' + width + ', ' + height);
    }
    canvas.width = width;
    canvas.height = height;
    // reset scale
    this.setScale({x: 1, y: 1, z: 1});
  };

  /**
   * Enable and listen to container interaction events.
   */
  this.bindInteraction = function () {
    // allow pointer events
    containerDiv.style.pointerEvents = 'auto';
    // interaction events
    var names = dwv.gui.interactionEventNames;
    for (var i = 0; i < names.length; ++i) {
      containerDiv.addEventListener(names[i], fireEvent);
    }
  };

  /**
   * Disable and stop listening to container interaction events.
   */
  this.unbindInteraction = function () {
    // disable pointer events
    containerDiv.style.pointerEvents = 'none';
    // interaction events
    var names = dwv.gui.interactionEventNames;
    for (var i = 0; i < names.length; ++i) {
      containerDiv.removeEventListener(names[i], fireEvent);
    }
  };

  /**
   * Add an event listener to this class.
   *
   * @param {string} type The event type.
   * @param {object} callback The method associated with the provided
   *   event type, will be called with the fired event.
   */
  this.addEventListener = function (type, callback) {
    listenerHandler.add(type, callback);
  };

  /**
   * Remove an event listener from this class.
   *
   * @param {string} type The event type.
   * @param {object} callback The method associated with the provided
   *   event type.
   */
  this.removeEventListener = function (type, callback) {
    listenerHandler.remove(type, callback);
  };

  /**
   * Fire an event: call all associated listeners with the input event object.
   *
   * @param {object} event The event to fire.
   * @private
   */
  function fireEvent(event) {
    event.srclayerid = self.getId();
    event.dataindex = dataIndex;
    listenerHandler.fireEvent(event);
  }

  // common layer methods [end] ---------------

  /**
   * Update the canvas image data.
   */
  function updateImageData() {
    // generate image data
    viewController.generateImageData(imageData);
    // pass the data to the off screen canvas
    offscreenCanvas.getContext('2d').putImageData(imageData, 0, 0);
    // update data flag
    needsDataUpdate = false;
  }

  /**
   * Handle window/level change.
   *
   * @param {object} event The event fired when changing the window/level.
   * @private
   */
  function onWLChange(event) {
    // generate and draw if no skip flag
    if (typeof event.skipGenerate === 'undefined' ||
      event.skipGenerate === false) {
      needsDataUpdate = true;
      self.draw();
    }
  }

  /**
   * Handle colour map change.
   *
   * @param {object} _event The event fired when changing the colour map.
   * @private
   */
  function onColourChange(_event) {
    needsDataUpdate = true;
    self.draw();
  }

  /**
   * Handle position change.
   *
   * @param {object} event The event fired when changing the position.
   * @private
   */
  function onPositionChange(event) {
    if (typeof event.skipGenerate === 'undefined' ||
      event.skipGenerate === false) {
      // clear for non valid events
      if (typeof event.valid !== 'undefined' && !event.valid) {
        self.clear();
        return;
      }
      // 3D dimensions
      var dims3D = [0, 1, 2];
      // remove scroll index
      var indexScrollIndex = dims3D.indexOf(viewController.getScrollIndex());
      dims3D.splice(indexScrollIndex, 1);
      // remove non scroll index from diff dims
      var diffDims = event.diffDims.filter(function (item) {
        return dims3D.indexOf(item) === -1;
      });
      // update if we have something left
      if (diffDims.length !== 0) {
        needsDataUpdate = true;
        self.draw();
      }
    }
  }

  /**
   * Handle alpha function change.
   *
   * @param {object} event The event fired when changing the function.
   * @private
   */
  function onAlphaFuncChange(event) {
    if (typeof event.skipGenerate === 'undefined' ||
      event.skipGenerate === false) {
      needsDataUpdate = true;
      self.draw();
    }
  }

  /**
   * Set the current position.
   *
   * @param {dwv.math.Point} position The new position.
   * @param {dwv.math.Index} _index The new index.
   */
  this.setCurrentPosition = function (position, _index) {
    viewController.setCurrentPosition(position);
  };

  /**
   * Clear the context.
   */
  this.clear = function () {
    if (!isContextClear) {
      // clear the context: reset the transform first
      // store the current transformation matrix
      context.save();
      // use the identity matrix while clearing the canvas
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      // restore the transform
      context.restore();

      // update clear flag
      isContextClear = true;
    }
  };

  /**
   * Align on another layer.
   *
   * @param {dwv.gui.ViewLayer} rhs The layer to align on.
   */
  this.align = function (rhs) {
    canvas.style.top = rhs.getCanvas().offsetTop;
    canvas.style.left = rhs.getCanvas().offsetLeft;
  };

}; // ViewLayer class
