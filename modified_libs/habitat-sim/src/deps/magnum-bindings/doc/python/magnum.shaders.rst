..
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020 Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
..

.. py:class:: magnum.shaders.Flat2D
    :data POSITION: Vertex position
    :data TEXTURE_COORDINATES: 2D texture coordinates
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:class:: magnum.shaders.Flat3D
    :data POSITION: Vertex position
    :data TEXTURE_COORDINATES: 2D texture coordinates
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:property:: magnum.shaders.Flat2D.alpha_mask
    :raise AttributeError: If the shader was not created with `Flags.ALPHA_MASK`
.. py:property:: magnum.shaders.Flat3D.alpha_mask
    :raise AttributeError: If the shader was not created with `Flags.ALPHA_MASK`

.. py:function:: magnum.shaders.Flat2D.bind_texture
    :raise AttributeError: If the shader was not created with `Flags.TEXTURED`
.. py:function:: magnum.shaders.Flat3D.bind_texture
    :raise AttributeError: If the shader was not created with `Flags.TEXTURED`

.. py:class:: magnum.shaders.VertexColor2D
    :data POSITION: Vertex position
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:class:: magnum.shaders.VertexColor3D
    :data POSITION: Vertex position
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:class:: magnum.shaders.Phong
    :data POSITION: Vertex position
    :data NORMAL: Normal direction
    :data TANGENT: Tangent direction
    :data TEXTURE_COORDINATES: 2D texture coordinates
    :data COLOR3: Three-component vertex color
    :data COLOR4: Four-component vertex color

.. py:property:: magnum.shaders.Phong.alpha_mask
    :raise AttributeError: If the shader was not created with `Flags.ALPHA_MASK`
.. py:property:: magnum.shaders.Phong.light_positions
    :raise ValueError: If list length is different from `light_count`
.. py:property:: magnum.shaders.Phong.light_colors
    :raise ValueError: If list length is different from `light_count`

.. py:function:: magnum.shaders.Phong.bind_ambient_texture
    :raise AttributeError: If the shader was not created with
        `Flags.AMBIENT_TEXTURE`
.. py:function:: magnum.shaders.Phong.bind_diffuse_texture
    :raise AttributeError: If the shader was not created with
        `Flags.DIFFUSE_TEXTURE`
.. py:function:: magnum.shaders.Phong.bind_specular_texture
    :raise AttributeError: If the shader was not created with
        `Flags.SPECULAR_TEXTURE`
.. py:function:: magnum.shaders.Phong.bind_normal_texture
    :raise AttributeError: If the shader was not created with
        `Flags.NORMAL_TEXTURE`
.. py:function:: magnum.shaders.Phong.bind_textures
    :raise AttributeError: If the shader was not created with any of
        `Flags.AMBIENT_TEXTURE`, `Flags.DIFFUSE_TEXTURE`,
        `Flags.SPECULAR_TEXTURE` or `Flags.NORMAL_TEXTURE`
