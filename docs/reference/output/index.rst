=========
Output
=========
.. currentmodule:: swmm.pandas

Constructor and Properties
---------------------------
.. autoclass:: Output

   .. autoattribute:: subcatch_attributes
   .. autoattribute:: link_attributes
   .. autoattribute:: node_attributes
   .. autoattribute:: system_attributes
   .. autoattribute:: subcatchments
   .. autoattribute:: links
   .. autoattribute:: nodes
   .. autoattribute:: pollutants
   .. autoattribute:: project_size
   .. autoattribute:: report
   .. autoattribute:: start
   .. autoattribute:: end
   .. autoattribute:: period
   .. autoattribute:: units

Time Series Data
-----------------
Get time series data for one or more elements and one or more variables.

.. autosummary::
   :nosignatures:
   :toctree: api/

   Output.link_series
   Output.node_series   
   Output.subcatch_series
   Output.system_series
   
Element Attribute Data
-----------------------
Get attribute for a given time step for all elements of a given type.

.. autosummary::
   :nosignatures:
   :toctree: api/

   Output.link_attribute
   Output.node_attribute   
   Output.subcatch_attribute
   Output.system_attribute

Element Result Data
--------------------
Get all attributes for a given set time steps for given set of elements.

.. autosummary::
   :nosignatures:
   :toctree: api/
   
   Output.link_result
   Output.node_result   
   Output.subcatch_result
   Output.system_result

Helper Methods
--------------

.. autosummary::
   :nosignatures:
   :toctree: api/

   Output.getStructure