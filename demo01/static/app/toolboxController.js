// namespaces
var dwv = dwv || {};
dwv.ctrl = dwv.ctrl || {};

/**
 * Toolbox controller.
 *
 * @param {Array} toolList The list of tool objects.
 * @class
 */
dwv.ctrl.ToolboxController = function (toolList) {
  /**
   * Selected tool.
   *
   * @type {object}
   * @private
   */
  var selectedTool = null;

  /**
   * Callback store to allow attach/detach.
   *
   * @type {Array}
   * @private
   */
  var callbackStore = [];

  /**
   * Initialise.
   */
  this.init = function () {
    for (var key in toolList) {
      toolList[key].init();
    }
    // keydown listener
    window.addEventListener('keydown', getOnMouch('window', 'keydown'), true);
  };

  /**
   * Get the tool list.
   *
   * @returns {Array} The list of tool objects.
   */
  this.getToolList = function () {
    return toolList;
  };

  /**
   * Check if a tool is in the tool list.
   *
   * @param {string} name The name to check.
   * @returns {string} The tool list element for the given name.
   */
  this.hasTool = function (name) {
    return typeof this.getToolList()[name] !== 'undefined';
  };

  /**
   * Get the selected tool.
   *
   * @returns {object} The selected tool.
   */
  this.getSelectedTool = function () {
    return selectedTool;
  };

  /**
   * Get the selected tool event handler.
   *
   * @param {string} eventType The event type, for example
   *   mousedown, touchstart...
   * @returns {Function} The event handler.
   */
  this.getSelectedToolEventHandler = function (eventType) {
    return this.getSelectedTool()[eventType];
  };

  /**
   * Set the selected tool.
   *
   * @param {string} name The name of the tool.
   */
  this.setSelectedTool = function (name) {
    // check if we have it
    if (!this.hasTool(name)) {
      throw new Error('Unknown tool: \'' + name + '\'');
    }
    // de-activate previous
    if (selectedTool) {
      selectedTool.activate(false);
    }
    // set internal var
    selectedTool = toolList[name];
    // activate new tool
    selectedTool.activate(true);
  };

  /**
   * Set the selected shape.
   *
   * @param {string} name The name of the shape.
   */
  this.setSelectedShape = function (name) {
    this.getSelectedTool().setShapeName(name);
  };

  /**
   * Set the selected filter.
   *
   * @param {string} name The name of the filter.
   */
  this.setSelectedFilter = function (name) {
    this.getSelectedTool().setSelectedFilter(name);
  };

  /**
   * Run the selected filter.
   */
  this.runSelectedFilter = function () {
    this.getSelectedTool().getSelectedFilter().run();
  };

  /**
   * Set the tool line colour.
   *
   * @param {string} colour The colour.
   */
  this.setLineColour = function (colour) {
    this.getSelectedTool().setLineColour(colour);
  };

  /**
   * Set the tool range.
   *
   * @param {object} range The new range of the data.
   */
  this.setRange = function (range) {
    // seems like jquery is checking if the method exists before it
    // is used...
    if (this.getSelectedTool() &&
      this.getSelectedTool().getSelectedFilter()) {
      this.getSelectedTool().getSelectedFilter().run(range);
    }
  };

  /**
   * Listen to layer interaction events.
   *
   * @param {object} layer The layer to listen to.
   */
  this.bindLayer = function (layer) {
    layer.bindInteraction();
    // interaction events
    var names = dwv.gui.interactionEventNames;
    for (var i = 0; i < names.length; ++i) {
      layer.addEventListener(names[i],
        getOnMouch(layer.getId(), names[i]));
    }
  };

  /**
   * Remove canvas mouse and touch listeners.
   *
   * @param {object} layer The layer to stop listening to.
   */
  this.unbindLayer = function (layer) {
    layer.unbindInteraction();
    // interaction events
    var names = dwv.gui.interactionEventNames;
    for (var i = 0; i < names.length; ++i) {
      layer.removeEventListener(names[i],
        getOnMouch(layer.getId(), names[i]));
    }
  };

  /**
   * Mou(se) and (T)ouch event handler. This function just determines
   * the mouse/touch position relative to the canvas element.
   * It then passes it to the current tool.
   *
   * @param {string} layerId The layer id.
   * @param {string} eventType The event type.
   * @returns {object} A callback for the provided layer and event.
   * @private
   */
  function getOnMouch(layerId, eventType) {
    // augment event with converted offsets
    var augmentEventOffsets = function (event) {
      // event offset(s)
      var offsets = dwv.gui.getEventOffset(event);
      // should have at least one offset
      event._x = offsets[0].x;
      event._y = offsets[0].y;
      // possible second
      if (offsets.length === 2) {
        event._x1 = offsets[1].x;
        event._y1 = offsets[1].y;
      }
    };

    var applySelectedTool = function (event) {
      // make sure we have a tool
      if (selectedTool) {
        var func = selectedTool[event.type];
        if (func) {
          func(event);
        }
      }
    };

    if (typeof callbackStore[layerId] === 'undefined') {
      callbackStore[layerId] = [];
    }

    if (typeof callbackStore[layerId][eventType] === 'undefined') {
      var callback = null;
      if (eventType === 'keydown') {
        callback = function (event) {
          applySelectedTool(event);
        };
      } else if (eventType === 'touchend') {
        callback = function (event) {
          event.preventDefault();
          applySelectedTool(event);
        };
      } else {
        // mouse or touch events
        callback = function (event) {
          event.preventDefault();
          augmentEventOffsets(event);
          applySelectedTool(event);
        };
      }
      // store callback
      callbackStore[layerId][eventType] = callback;
    }

    return callbackStore[layerId][eventType];
  }

}; // class ToolboxController
